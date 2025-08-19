import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:uuid/uuid.dart';

import '../models/message.dart';
import '../providers/chat_notifier.dart';
import '../theme/app_theme.dart';
import '../welcome_section.dart';
import 'quick_replies.dart';
import 'chat/index.dart';

/// A scalable and modular AI chat widget built with reusable components.
/// 
/// This widget provides a complete chat interface with:
/// - Modular, reusable widget components
/// - Word-by-word streaming text animations for AI responses
/// - Customizable message bubbles with modern styling
/// - Smooth scroll behavior and auto-scroll
/// - Modern input field with glassmorphic design
/// - Typing indicators with smooth animations
/// - Message editing support
/// - Attachment handling capabilities
/// - Responsive design and cross-platform compatibility
/// 
/// The widget uses modular components from the chat/ directory for
/// better maintainability, reusability, and scalability.
class AiChatWidget extends ConsumerStatefulWidget {
  /// Optional session ID. If not provided, a new UUID will be generated.
  final String? sessionId;
  
  /// Optional welcome prompts to show initially.
  final bool showWelcomePrompts;
  
  /// Optional quick replies to show initially.
  final bool showQuickReplies;
  
  /// Optional custom padding for the chat interface.
  final EdgeInsetsGeometry? padding;
  
  /// Optional custom height for the widget.
  final double? height;

  const AiChatWidget({
    super.key,
    this.sessionId,
    this.showWelcomePrompts = true,
    this.showQuickReplies = true,
    this.padding,
    this.height,
  });

  @override
  ConsumerState<AiChatWidget> createState() => _AiChatWidgetState();
}

class _AiChatWidgetState extends ConsumerState<AiChatWidget>
    with TickerProviderStateMixin {
  late final ScrollController _scrollController;
  late final TextEditingController _textController;
  late final String _sessionId;
  late final AnimationController _typingAnimationController;
  late final AnimationController _fadeAnimationController;
  
  @override
  void initState() {
    super.initState();
    
    _scrollController = ScrollController();
    _textController = TextEditingController();
    _sessionId = widget.sessionId ?? const Uuid().v4();
    
    _typingAnimationController = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    )..repeat();
    
    _fadeAnimationController = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    )..forward();
    
    // Listen to chat state changes to auto-scroll
    WidgetsBinding.instance.addPostFrameCallback((_) {
      // Auto-scroll is now handled through Riverpod state listening
      _scrollToBottom();
    });
  }

  @override
  void dispose() {
    _scrollController.dispose();
    _textController.dispose();
    _typingAnimationController.dispose();
    _fadeAnimationController.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    if (_scrollController.hasClients) {
      Future.delayed(const Duration(milliseconds: 100), () {
        if (_scrollController.hasClients) {
          _scrollController.animateTo(
            _scrollController.position.maxScrollExtent,
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeOutCubic,
          );
        }
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final chatState = ref.watch(chatNotifierProvider);
    
    // Auto-scroll when messages change
    ref.listen(chatNotifierProvider, (previous, next) {
      if (previous?.messages.length != next.messages.length) {
        WidgetsBinding.instance.addPostFrameCallback((_) {
          _scrollToBottom();
        });
      }
    });
    
    return Container(
      height: widget.height,
      padding: widget.padding,
      child: Column(
        children: [
          // Messages area
          Expanded(
            child: Builder(
              builder: (context) {
                return Container(
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.topCenter,
                      end: Alignment.bottomCenter,
                      colors: [
                        Theme.of(context).scaffoldBackgroundColor,
                        Theme.of(context).scaffoldBackgroundColor.withValues(alpha: 0.95),
                      ],
                    ),
                  ),
                  child: CustomScrollView(
                    controller: _scrollController,
                    physics: const BouncingScrollPhysics(),
                    slivers: [
                      SliverPadding(
                        padding: const EdgeInsets.all(16),
                        sliver: SliverList(
                          delegate: SliverChildListDelegate([
                            // Welcome prompts
                            if (widget.showWelcomePrompts && 
                                chatState.showWelcomePrompts && 
                                chatState.messages.length == 1)
                              FadeTransition(
                                opacity: _fadeAnimationController,
                                child: WelcomeSection(sessionId: _sessionId),
                              ),

                            // Messages with enhanced animations
                            ..._buildAnimatedMessages(chatState.messages),

                            // Enhanced typing indicator using new component
                            if (chatState.isTyping)
                              FadeTransition(
                                opacity: _fadeAnimationController,
                                child: TypingIndicator(
                                  isVisible: chatState.isTyping,
                                  message: 'Fitvise AI is thinking...',
                                ),
                              ),

                            // Quick replies
                            if (widget.showQuickReplies && 
                                chatState.showQuickReplies && 
                                chatState.messages.length == 1)
                              FadeTransition(
                                opacity: _fadeAnimationController,
                                child: QuickReplies(sessionId: _sessionId),
                              ),

                            const SizedBox(height: 16),
                          ]),
                        ),
                      ),
                    ],
                  ),
                );
              },
            ),
          ),

          // Enhanced input area using new component
          _buildModularInputArea(),
        ],
      ),
    );
  }

  List<Widget> _buildAnimatedMessages(List<Message> messages) {
    return messages.asMap().entries.map((entry) {
      final index = entry.key;
      final message = entry.value;
      
      return AnimatedContainer(
        duration: Duration(milliseconds: 300 + (index * 50)),
        curve: Curves.easeOutCubic,
        margin: const EdgeInsets.symmetric(vertical: 4),
        child: SlideTransition(
          position: Tween<Offset>(
            begin: Offset(message.sender == 'user' ? 1.0 : -1.0, 0.0),
            end: Offset.zero,
          ).animate(CurvedAnimation(
            parent: _fadeAnimationController,
            curve: Interval(
              (index * 0.1).clamp(0.0, 1.0),
              ((index * 0.1) + 0.3).clamp(0.0, 1.0),
              curve: Curves.easeOutCubic,
            ),
          )),
          child: FadeTransition(
            opacity: CurvedAnimation(
              parent: _fadeAnimationController,
              curve: Interval(
                (index * 0.1).clamp(0.0, 1.0),
                ((index * 0.1) + 0.3).clamp(0.0, 1.0),
                curve: Curves.easeOut,
              ),
            ),
            child: _buildModularMessageBubble(message),
          ),
        ),
      );
    }).toList();
  }

  Widget _buildModularMessageBubble(Message message) {
    final chatState = ref.watch(chatNotifierProvider);
    final chatNotifier = ref.read(chatNotifierProvider.notifier);
    final isStreaming = chatState.streamingMessageId == message.id;
    final isUser = message.sender == 'user';
    
    // Handle editing mode
    if (chatState.editingMessageId == message.id) {
      return _buildEnhancedEditingInterface(chatState, chatNotifier);
    }

    return MessageBubble(
      text: message.text,
      isUser: isUser,
      timestamp: message.timestamp,
      isEdited: message.isEdited,
      senderName: isUser ? null : 'Fitvise AI',
      content: isStreaming ? AnimatedTextMessage(
        text: message.text,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 14.5,
          height: 1.5,
          fontWeight: FontWeight.w400,
        ),
        wordDelay: const Duration(milliseconds: 100),
        showCursor: true,
      ) : null,
      actions: message.actions != null && message.actions!.isNotEmpty
          ? message.actions!.map((action) {
              return _buildEnhancedActionButton(action);
            }).toList()
          : null,
      messageActions: [
        _buildActionButton(
          icon: Icons.copy_rounded,
          onPressed: () => _copyToClipboard(message.text),
          tooltip: 'Copy message',
        ),
        if (isUser)
          _buildActionButton(
            icon: Icons.edit_rounded,
            onPressed: () => chatNotifier.startEditingMessage(message.id, message.text),
            tooltip: 'Edit message',
          ),
      ],
      animated: true,
      animationDuration: Duration(milliseconds: 300 + (chatState.messages.indexOf(message) * 50.0).toInt()),
    );
  }

  // Legacy message bubble method removed - now using MessageBubble component

  // Legacy message content method removed - now using MessageBubble component

  // Old streaming text method removed - now using AnimatedTextMessage component

  // Old typing indicator methods removed - now using TypingIndicator component

  // Legacy message actions method removed - now handled by MessageBubble component

  Widget _buildActionButton({
    required IconData icon,
    required VoidCallback onPressed,
    required String tooltip,
  }) {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 2),
      child: Material(
        color: Theme.of(context).brightness == Brightness.dark
            ? const Color(0xFF374151).withValues(alpha: 0.9)
            : Colors.white.withValues(alpha: 0.95),
        borderRadius: BorderRadius.circular(18),
        elevation: 2,
        shadowColor: Colors.black.withValues(alpha: 0.1),
        child: InkWell(
          borderRadius: BorderRadius.circular(18),
          onTap: onPressed,
          child: Container(
            padding: const EdgeInsets.all(8),
            child: Icon(
              icon,
              size: 16,
              color: Theme.of(context).brightness == Brightness.dark
                  ? Colors.white.withValues(alpha: 0.8)
                  : const Color(0xFF374151).withValues(alpha: 0.8),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildEnhancedActionButton(MessageAction action) {
    final chatNotifier = ref.read(chatNotifierProvider.notifier);
    
    return AnimatedContainer(
      duration: const Duration(milliseconds: 200),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(24),
          onTap: () => chatNotifier.sendMessage(_sessionId, action.label),
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [
                  AppTheme.primaryBlue.withValues(alpha: 0.1),
                  AppTheme.secondaryPurple.withValues(alpha: 0.1),
                ],
              ),
              borderRadius: BorderRadius.circular(24),
              border: Border.all(
                color: AppTheme.primaryBlue.withValues(alpha: 0.3),
              ),
            ),
            child: Text(
              action.label,
              style: TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w500,
                color: AppTheme.primaryBlue,
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildEnhancedEditingInterface(ChatState chatState, ChatNotifier chatNotifier) {
    return Container(
      constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.78),
      margin: const EdgeInsets.only(right: 8),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Theme.of(context).brightness == Brightness.dark
            ? const Color(0xFF374151).withValues(alpha: 0.9)
            : Colors.white.withValues(alpha: 0.95),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: AppTheme.primaryBlue.withValues(alpha: 0.3),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.1),
            blurRadius: 16,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        children: [
          TextFormField(
            initialValue: chatState.editText,
            onChanged: chatNotifier.updateEditText,
            maxLines: 3,
            style: TextStyle(
              color: Theme.of(context).brightness == Brightness.dark
                  ? Colors.white
                  : const Color(0xFF111827),
              fontSize: 14.5,
            ),
            decoration: InputDecoration(
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(16),
                borderSide: BorderSide(
                  color: Theme.of(context).dividerColor.withValues(alpha: 0.3),
                ),
              ),
              enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(16),
                borderSide: BorderSide(
                  color: Theme.of(context).dividerColor.withValues(alpha: 0.3),
                ),
              ),
              focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(16),
                borderSide: BorderSide(
                  color: AppTheme.primaryBlue,
                  width: 2,
                ),
              ),
              hintText: 'Edit your message...',
              hintStyle: TextStyle(
                color: Theme.of(context).textTheme.bodySmall?.color?.withValues(alpha: 0.5),
              ),
              contentPadding: const EdgeInsets.all(16),
            ),
            autofocus: true,
          ),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              TextButton(
                onPressed: chatNotifier.cancelEditing,
                style: TextButton.styleFrom(
                  foregroundColor: Theme.of(context).textTheme.bodyMedium?.color,
                ),
                child: const Text('Cancel'),
              ),
              const SizedBox(width: 12),
              ElevatedButton(
                onPressed: () {
                  chatNotifier.sendMessage(
                    _sessionId,
                    chatState.editText,
                    isEdit: true,
                    editId: chatState.editingMessageId,
                  );
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.primaryBlue,
                  foregroundColor: Colors.white,
                  elevation: 2,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: const Text('Save'),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildModularInputArea() {
    return Builder(
      builder: (context) {
        final chatState = ref.watch(chatNotifierProvider);
        final chatNotifier = ref.read(chatNotifierProvider.notifier);
        
        return ChatInput(
          controller: _textController,
          onSend: (text) {
            chatNotifier.sendMessage(_sessionId, text);
            _textController.clear();
          },
          onVoiceToggle: chatNotifier.toggleRecording,
          isRecording: chatState.isRecording,
          isLoading: chatState.isLoading,
          statusText: _getStatusText(chatState),
          config: const ChatInputConfig(
            hintText: 'Ask about workouts, nutrition, or fitness goals...',
            maxLength: 2000,
            enableVoiceInput: true,
            enableAttachments: true,
            showCharacterCount: true,
            showStatusText: true,
            attachmentOptions: [
              AttachmentOption(
                icon: Icons.image_rounded,
                tooltip: 'Upload workout photo',
                type: 'image',
              ),
              AttachmentOption(
                icon: Icons.attach_file_rounded,
                tooltip: 'Upload file',
                type: 'file',
              ),
              AttachmentOption(
                icon: Icons.description_rounded,
                tooltip: 'Upload fitness plan',
                type: 'document',
              ),
            ],
          ),
        );
      },
    );
  }

  // Legacy helper methods removed - now using modular components

  // Legacy send message method removed - handled by ChatInput component


  // File upload handling moved to ChatInput component

  void _copyToClipboard(String text) {
    Clipboard.setData(ClipboardData(text: text));
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Message copied to clipboard'),
        duration: Duration(seconds: 2),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  String _getStatusText(ChatState chatState) {
    if (chatState.isRecording) {
      return '🔴 Recording... Tap mic to stop';
    } else if (chatState.isTyping) {
      return 'Fitvise AI is thinking...';
    }
    return 'Shift+Enter for new line';
  }

  // Legacy time formatting moved to MessageBubble component
}