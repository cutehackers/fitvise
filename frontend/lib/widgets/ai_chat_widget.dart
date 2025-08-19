import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:uuid/uuid.dart';

import '../models/message.dart';
import '../providers/chat_provider.dart';
import '../theme/app_theme.dart';
import '../welcome_section.dart';
import 'quick_replies.dart';

/// A comprehensive AI chat widget that implements flutter_gen_ai_chat_ui functionality.
/// 
/// This widget provides a complete chat interface with:
/// - Word-by-word streaming text animations for AI responses
/// - Customizable message bubbles with modern styling
/// - Smooth scroll behavior and auto-scroll
/// - Modern input field with glassmorphic design
/// - Typing indicators with smooth animations
/// - Message editing support
/// - Responsive design and cross-platform compatibility
/// 
/// The widget integrates seamlessly with the existing Message model and ChatProvider
/// while maintaining all existing functionality.
class AiChatWidget extends StatefulWidget {
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
  State<AiChatWidget> createState() => _AiChatWidgetState();
}

class _AiChatWidgetState extends State<AiChatWidget>
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
    
    // Listen to chat provider changes to auto-scroll
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final chatProvider = Provider.of<ChatProvider>(context, listen: false);
      chatProvider.addListener(_scrollToBottom);
    });
  }

  @override
  void dispose() {
    final chatProvider = Provider.of<ChatProvider>(context, listen: false);
    chatProvider.removeListener(_scrollToBottom);
    
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
    return Container(
      height: widget.height,
      padding: widget.padding,
      child: Column(
        children: [
          // Messages area
          Expanded(
            child: Consumer<ChatProvider>(
              builder: (context, chatProvider, child) {
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
                                chatProvider.showWelcomePrompts && 
                                chatProvider.messages.length == 1)
                              FadeTransition(
                                opacity: _fadeAnimationController,
                                child: WelcomeSection(sessionId: _sessionId),
                              ),

                            // Messages with enhanced animations
                            ..._buildAnimatedMessages(chatProvider.messages),

                            // Enhanced typing indicator
                            if (chatProvider.isTyping)
                              FadeTransition(
                                opacity: _fadeAnimationController,
                                child: _buildEnhancedTypingIndicator(),
                              ),

                            // Quick replies
                            if (widget.showQuickReplies && 
                                chatProvider.showQuickReplies && 
                                chatProvider.messages.length == 1)
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

          // Enhanced input area
          _buildEnhancedInputArea(),
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
            child: _buildEnhancedMessageBubble(message),
          ),
        ),
      );
    }).toList();
  }

  Widget _buildEnhancedMessageBubble(Message message) {
    final isUser = message.sender == 'user';
    final chatProvider = Provider.of<ChatProvider>(context, listen: false);
    final isStreaming = chatProvider.streamingMessageId == message.id;

    return Container(
      margin: const EdgeInsets.symmetric(vertical: 4),
      child: Column(
        crossAxisAlignment: isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
        children: [
          // AI avatar and name with enhanced styling
          if (!isUser)
            Padding(
              padding: const EdgeInsets.only(left: 8, bottom: 6),
              child: Row(
                children: [
                  Container(
                    width: 28,
                    height: 28,
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(
                        colors: [AppTheme.primaryBlue, AppTheme.secondaryPurple],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      borderRadius: BorderRadius.circular(14),
                      boxShadow: [
                        BoxShadow(
                          color: AppTheme.primaryBlue.withValues(alpha: 0.3),
                          blurRadius: 8,
                          offset: const Offset(0, 2),
                        ),
                      ],
                    ),
                    child: const Icon(
                      Icons.fitness_center,
                      color: Colors.white,
                      size: 14,
                    ),
                  ),
                  const SizedBox(width: 10),
                  Text(
                    'Fitvise AI',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: Theme.of(context).textTheme.bodySmall?.color?.withValues(alpha: 0.7),
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ),

          // Enhanced message content
          Consumer<ChatProvider>(
            builder: (context, chatProvider, child) {
              if (chatProvider.editingMessageId == message.id) {
                return _buildEnhancedEditingInterface(chatProvider);
              }
              return _buildEnhancedMessageContent(message, isStreaming);
            },
          ),

          // Action buttons for AI messages
          if (!isUser && message.actions != null && message.actions!.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(left: 38, top: 12),
              child: Wrap(
                spacing: 8,
                runSpacing: 8,
                children: message.actions!.map((action) {
                  return _buildEnhancedActionButton(action);
                }).toList(),
              ),
            ),

          // Enhanced timestamp
          Padding(
            padding: EdgeInsets.only(
              left: isUser ? 0 : 38,
              right: isUser ? 8 : 0,
              top: 6,
            ),
            child: Text(
              _formatTime(message.timestamp),
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: Theme.of(context).textTheme.bodySmall?.color?.withValues(alpha: 0.5),
                fontSize: 11,
                fontWeight: FontWeight.w400,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildEnhancedMessageContent(Message message, bool isStreaming) {
    final isUser = message.sender == 'user';

    return Row(
      mainAxisAlignment: isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Flexible(
          child: Container(
            constraints: BoxConstraints(
              maxWidth: MediaQuery.of(context).size.width * 0.78,
            ),
            child: Stack(
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
                  margin: EdgeInsets.only(left: isUser ? 0 : 10),
                  decoration: BoxDecoration(
                    gradient: isUser
                        ? const LinearGradient(
                            colors: [AppTheme.primaryBlue, AppTheme.secondaryPurple],
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                          )
                        : null,
                    color: !isUser
                        ? (Theme.of(context).brightness == Brightness.dark
                            ? const Color(0xFF374151).withValues(alpha: 0.8)
                            : Colors.white.withValues(alpha: 0.9))
                        : null,
                    borderRadius: BorderRadius.circular(22),
                    border: !isUser
                        ? Border.all(
                            color: Theme.of(context).brightness == Brightness.dark
                                ? const Color(0xFF4B5563).withValues(alpha: 0.5)
                                : const Color(0xFFE5E7EB).withValues(alpha: 0.8),
                          )
                        : null,
                    boxShadow: [
                      BoxShadow(
                        color: (isUser ? AppTheme.primaryBlue : Colors.black)
                            .withValues(alpha: 0.1),
                        blurRadius: 12,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Streaming text with word-by-word animation
                      isStreaming
                          ? _buildStreamingText(message.text)
                          : Text(
                              message.text,
                              style: TextStyle(
                                color: isUser
                                    ? Colors.white
                                    : (Theme.of(context).brightness == Brightness.dark
                                        ? Colors.white.withValues(alpha: 0.95)
                                        : const Color(0xFF111827)),
                                fontSize: 14.5,
                                height: 1.5,
                                fontWeight: FontWeight.w400,
                              ),
                            ),
                      if (message.isEdited)
                        Padding(
                          padding: const EdgeInsets.only(top: 6),
                          child: Text(
                            'edited',
                            style: TextStyle(
                              color: isUser
                                  ? Colors.white.withValues(alpha: 0.7)
                                  : Theme.of(context)
                                      .textTheme
                                      .bodySmall
                                      ?.color
                                      ?.withValues(alpha: 0.5),
                              fontSize: 11,
                              fontStyle: FontStyle.italic,
                              fontWeight: FontWeight.w300,
                            ),
                          ),
                        ),
                    ],
                  ),
                ),

                // Enhanced message actions
                Positioned(
                  right: isUser ? -10 : null,
                  left: !isUser ? -10 : null,
                  top: 0,
                  bottom: 0,
                  child: _buildEnhancedMessageActions(message, isUser),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildStreamingText(String text) {
    if (text.isEmpty) return const SizedBox.shrink();
    
    final words = text.split(' ');
    return Wrap(
      children: words.asMap().entries.map((entry) {
        final index = entry.key;
        final word = entry.value;
        
        return AnimatedOpacity(
          duration: Duration(milliseconds: 100 + (index * 50)),
          opacity: 1.0,
          child: AnimatedContainer(
            duration: Duration(milliseconds: 200 + (index * 30)),
            curve: Curves.easeOutCubic,
            child: Text(
              '$word ',
              style: const TextStyle(
                color: Colors.white,
                fontSize: 14.5,
                height: 1.5,
                fontWeight: FontWeight.w400,
              ),
            ),
          ),
        );
      }).toList(),
    );
  }

  Widget _buildEnhancedTypingIndicator() {
    return Container(
      alignment: Alignment.centerLeft,
      padding: const EdgeInsets.symmetric(vertical: 10),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
        margin: const EdgeInsets.only(left: 10),
        decoration: BoxDecoration(
          color: Theme.of(context).brightness == Brightness.dark
              ? const Color(0xFF374151).withValues(alpha: 0.8)
              : Colors.white.withValues(alpha: 0.9),
          borderRadius: BorderRadius.circular(22),
          border: Border.all(
            color: Theme.of(context).brightness == Brightness.dark
                ? const Color(0xFF4B5563).withValues(alpha: 0.5)
                : const Color(0xFFE5E7EB).withValues(alpha: 0.8),
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.08),
              blurRadius: 12,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          spacing: 6,
          children: [
            _buildEnhancedTypingDot(0),
            _buildEnhancedTypingDot(200),
            _buildEnhancedTypingDot(400),
          ],
        ),
      ),
    );
  }

  Widget _buildEnhancedTypingDot(int delay) {
    return AnimatedBuilder(
      animation: _typingAnimationController,
      builder: (context, child) {
        final animationValue = Curves.easeInOut.transform(
          (_typingAnimationController.value + (delay / 600)) % 1.0,
        );
        final scale = 0.7 + (0.4 * (0.5 + 0.5 * animationValue));
        final opacity = 0.3 + (0.7 * animationValue);
        
        return Transform.scale(
          scale: scale,
          child: Container(
            width: 10,
            height: 10,
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [
                  AppTheme.primaryBlue.withValues(alpha: opacity),
                  AppTheme.secondaryPurple.withValues(alpha: opacity),
                ],
              ),
              shape: BoxShape.circle,
            ),
          ),
        );
      },
    );
  }

  Widget _buildEnhancedMessageActions(Message message, bool isUser) {
    final chatProvider = Provider.of<ChatProvider>(context, listen: false);
    
    return SizedBox(
      width: 90,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // Copy button
          _buildActionButton(
            icon: Icons.copy_rounded,
            onPressed: () => _copyToClipboard(message.text),
            tooltip: 'Copy message',
          ),

          if (isUser) ...[
            const SizedBox(height: 4),
            _buildActionButton(
              icon: Icons.edit_rounded,
              onPressed: () => chatProvider.startEditingMessage(message.id, message.text),
              tooltip: 'Edit message',
            ),
          ],
        ],
      ),
    );
  }

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
    final chatProvider = Provider.of<ChatProvider>(context, listen: false);
    
    return AnimatedContainer(
      duration: const Duration(milliseconds: 200),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(24),
          onTap: () => chatProvider.sendMessage(_sessionId, action.label),
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

  Widget _buildEnhancedEditingInterface(ChatProvider chatProvider) {
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
            initialValue: chatProvider.editText,
            onChanged: chatProvider.updateEditText,
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
                onPressed: chatProvider.cancelEditing,
                style: TextButton.styleFrom(
                  foregroundColor: Theme.of(context).textTheme.bodyMedium?.color,
                ),
                child: const Text('Cancel'),
              ),
              const SizedBox(width: 12),
              ElevatedButton(
                onPressed: () {
                  chatProvider.sendMessage(
                    _sessionId,
                    chatProvider.editText,
                    isEdit: true,
                    editId: chatProvider.editingMessageId,
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

  Widget _buildEnhancedInputArea() {
    return Consumer<ChatProvider>(
      builder: (context, chatProvider, child) {
        return Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: Theme.of(context).scaffoldBackgroundColor.withValues(alpha: 0.95),
            border: Border(
              top: BorderSide(
                color: Theme.of(context).dividerColor.withValues(alpha: 0.3),
              ),
            ),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.05),
                blurRadius: 20,
                offset: const Offset(0, -4),
              ),
            ],
          ),
          child: Column(
            children: [
              // Attachment options
              Row(
                children: [
                  _buildAttachmentButton(
                    icon: Icons.image_rounded,
                    tooltip: 'Upload workout photo',
                    onPressed: () => _handleFileUpload('image'),
                  ),
                  const SizedBox(width: 10),
                  _buildAttachmentButton(
                    icon: Icons.attach_file_rounded,
                    tooltip: 'Upload file',
                    onPressed: () => _handleFileUpload('file'),
                  ),
                  const SizedBox(width: 10),
                  _buildAttachmentButton(
                    icon: Icons.description_rounded,
                    tooltip: 'Upload fitness plan',
                    onPressed: () => _handleFileUpload('document'),
                  ),
                ],
              ),
              const SizedBox(height: 16),

              // Enhanced input field
              Container(
                padding: const EdgeInsets.all(4),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      AppTheme.primaryBlue.withValues(alpha: 0.1),
                      AppTheme.secondaryPurple.withValues(alpha: 0.1),
                    ],
                  ),
                  borderRadius: BorderRadius.circular(28),
                  border: Border.all(
                    color: AppTheme.primaryBlue.withValues(alpha: 0.2),
                    width: 1.5,
                  ),
                ),
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  decoration: BoxDecoration(
                    color: Theme.of(context).brightness == Brightness.dark
                        ? const Color(0xFF374151).withValues(alpha: 0.9)
                        : Colors.white.withValues(alpha: 0.95),
                    borderRadius: BorderRadius.circular(24),
                  ),
                  child: Row(
                    children: [
                      // Text input
                      Expanded(
                        child: TextField(
                          controller: _textController,
                          maxLines: null,
                          textCapitalization: TextCapitalization.sentences,
                          textInputAction: TextInputAction.send,
                          style: TextStyle(
                            color: Theme.of(context).brightness == Brightness.dark
                                ? Colors.white
                                : const Color(0xFF111827),
                            fontSize: 15,
                            height: 1.4,
                          ),
                          decoration: InputDecoration(
                            hintText: 'Ask about workouts, nutrition, or fitness goals...',
                            hintStyle: TextStyle(
                              color: Theme.of(context).textTheme.bodySmall?.color?.withValues(alpha: 0.6),
                              fontSize: 15,
                            ),
                            border: InputBorder.none,
                            contentPadding: EdgeInsets.zero,
                          ),
                          onSubmitted: (text) {
                            if (text.trim().isNotEmpty) {
                              _sendMessage(chatProvider);
                            }
                          },
                        ),
                      ),
                      const SizedBox(width: 12),

                      // Voice recording button
                      _buildVoiceButton(chatProvider),
                      const SizedBox(width: 8),

                      // Enhanced send button
                      _buildEnhancedSendButton(chatProvider),
                    ],
                  ),
                ),
              ),

              // Status and character count
              Padding(
                padding: const EdgeInsets.only(top: 12),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    // Status text
                    Expanded(
                      child: Text(
                        _getStatusText(chatProvider),
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: Theme.of(context).textTheme.bodySmall?.color?.withValues(alpha: 0.6),
                          fontSize: 12,
                        ),
                      ),
                    ),

                    // Character count
                    Text(
                      '${_textController.text.length}/2000',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: Theme.of(context).textTheme.bodySmall?.color?.withValues(alpha: 0.6),
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildAttachmentButton({
    required IconData icon,
    required String tooltip,
    required VoidCallback onPressed,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: Theme.of(context).brightness == Brightness.dark
            ? const Color(0xFF374151).withValues(alpha: 0.5)
            : Colors.white.withValues(alpha: 0.8),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: Theme.of(context).dividerColor.withValues(alpha: 0.3),
        ),
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(12),
          onTap: onPressed,
          child: Padding(
            padding: const EdgeInsets.all(8),
            child: Icon(
              icon,
              size: 20,
              color: Theme.of(context).iconTheme.color?.withValues(alpha: 0.7),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildVoiceButton(ChatProvider chatProvider) {
    return Container(
      decoration: BoxDecoration(
        color: chatProvider.isRecording
            ? Colors.red.withValues(alpha: 0.1)
            : Colors.transparent,
        borderRadius: BorderRadius.circular(20),
        border: chatProvider.isRecording
            ? Border.all(color: Colors.red.withValues(alpha: 0.3))
            : null,
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(20),
          onTap: chatProvider.toggleRecording,
          child: Padding(
            padding: const EdgeInsets.all(8),
            child: Icon(
              chatProvider.isRecording ? Icons.mic_off_rounded : Icons.mic_rounded,
              size: 20,
              color: chatProvider.isRecording
                  ? Colors.red
                  : Theme.of(context).iconTheme.color?.withValues(alpha: 0.7),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildEnhancedSendButton(ChatProvider chatProvider) {
    final canSend = _textController.text.trim().isNotEmpty && !chatProvider.isLoading;
    
    return AnimatedContainer(
      duration: const Duration(milliseconds: 200),
      decoration: BoxDecoration(
        gradient: canSend
            ? const LinearGradient(
                colors: [AppTheme.primaryBlue, AppTheme.secondaryPurple],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              )
            : null,
        color: !canSend
            ? Theme.of(context).disabledColor.withValues(alpha: 0.3)
            : null,
        borderRadius: BorderRadius.circular(20),
        boxShadow: canSend
            ? [
                BoxShadow(
                  color: AppTheme.primaryBlue.withValues(alpha: 0.3),
                  blurRadius: 8,
                  offset: const Offset(0, 2),
                ),
              ]
            : null,
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(20),
          onTap: canSend ? () => _sendMessage(chatProvider) : null,
          child: Padding(
            padding: const EdgeInsets.all(10),
            child: chatProvider.isLoading
                ? SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      valueColor: const AlwaysStoppedAnimation<Color>(Colors.white),
                    ),
                  )
                : Icon(
                    Icons.send_rounded,
                    size: 16,
                    color: canSend ? Colors.white : Theme.of(context).disabledColor,
                  ),
          ),
        ),
      ),
    );
  }

  void _sendMessage(ChatProvider chatProvider) {
    final text = _textController.text.trim();
    if (text.isNotEmpty) {
      chatProvider.sendMessage(_sessionId, text);
      _textController.clear();
    }
  }

  void _handleFileUpload(String type) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('File upload ($type) coming soon!'),
        duration: const Duration(seconds: 2),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    );
  }

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

  String _getStatusText(ChatProvider chatProvider) {
    if (chatProvider.isRecording) {
      return '🔴 Recording... Tap mic to stop';
    } else if (chatProvider.isTyping) {
      return 'Fitvise AI is thinking...';
    }
    return 'Shift+Enter for new line';
  }

  String _formatTime(DateTime timestamp) {
    final hour = timestamp.hour > 12 ? timestamp.hour - 12 : (timestamp.hour == 0 ? 12 : timestamp.hour);
    final period = timestamp.hour >= 12 ? 'PM' : 'AM';
    final minute = timestamp.minute.toString().padLeft(2, '0');
    return '$hour:$minute $period';
  }
}