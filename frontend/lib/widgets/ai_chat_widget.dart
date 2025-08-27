import 'package:fitvise/providers/message_ids_provider.dart';
import 'package:fitvise/providers/message_provider.dart';
import 'package:fitvise/widgets/chat/message_bubble.dart';
import 'package:fitvise/widgets/chat/message_composer.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:gap/gap.dart';
import 'package:uuid/uuid.dart';

import '../models/message.dart';
import '../providers/message_list_provider.dart';
import '../theme/app_theme.dart';

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

class _AiChatWidgetState extends ConsumerState<AiChatWidget> with TickerProviderStateMixin {
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

    _typingAnimationController = AnimationController(duration: const Duration(milliseconds: 1500), vsync: this)
      ..repeat();

    _fadeAnimationController = AnimationController(duration: const Duration(milliseconds: 300), vsync: this)..forward();

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
    //final isTyping = ref.watch(messageListProvider.select((s) => s.isTyping));

    final messageIds = ref.watch(messageIdsProvider);

    if (messageIds.isNotEmpty) {
      ref.listen(messageProvider(messageIds.last), (_, _) {
        _scrollToBottom();
      });
    }

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
                            // if (widget.showWelcomePrompts &&
                            //     chatState.showWelcomePrompts &&
                            //     chatState.messages.length == 1)
                            //   FadeTransition(
                            //     opacity: _fadeAnimationController,
                            //     child: WelcomeSection(sessionId: _sessionId),
                            //   ),

                            // Messages with enhanced animations
                            ..._buildAnimatedMessages(messageIds),

                            // Enhanced typing indicator using new component
                            // if (isTyping)
                            //   FadeTransition(
                            //     opacity: _fadeAnimationController,
                            //     child: TypingIndicator(isVisible: isTyping, message: 'thinking...'),
                            //   ),

                            // Quick replies
                            // if (widget.showQuickReplies && chatState.showQuickReplies && chatState.messages.length == 1)
                            //   FadeTransition(
                            //     opacity: _fadeAnimationController,
                            //     child: QuickReplies(sessionId: _sessionId),
                            //   ),

                            // space
                            const Gap(16),
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
          Consumer(
            builder: (context, ref, child) {
              final isLoading = ref.watch(messageListProvider.select((e) => e.isLoading));
              final isRecording = ref.watch(messageListProvider.select((e) => e.isRecording));
              final isStreaming = ref.watch(messageListProvider.select((e) => e.isStreaming));
              final chatNotifier = ref.read(messageListProvider.notifier);

              return MessageComposer(
                controller: _textController,
                onSend: (text) {
                  chatNotifier.sendMessage(_sessionId, text);
                  _textController.clear();
                },
                onVoiceToggle: chatNotifier.toggleRecording,
                isRecording: isRecording,
                isLoading: isLoading,
                statusText: _getStatusText(isRecording, isStreaming),
                hintText: 'Ask about workouts, nutrition, or fitness goals...',
                maxLength: 2000,
                enableVoiceInput: true,
                enableAttachments: true,
                showCharacterCount: true,
                showStatusText: true,
                attachmentOptions: const [
                  AttachmentOption(icon: Icons.image_rounded, tooltip: 'Upload workout photo', type: 'image'),
                  AttachmentOption(icon: Icons.attach_file_rounded, tooltip: 'Upload file', type: 'file'),
                  AttachmentOption(icon: Icons.description_rounded, tooltip: 'Upload fitness plan', type: 'document'),
                ],
              );
            },
          ),
        ],
      ),
    );
  }

  List<Widget> _buildAnimatedMessages(List<String> messageIds) {
    return messageIds.asMap().entries.map((entry) {
      final index = entry.key;
      final messageId = entry.value;

      return Consumer(
        builder: (context, ref, child) {
          final message = ref.watch(messageProvider(messageId));

          return AnimatedContainer(
            key: ValueKey(messageId),
            duration: Duration(milliseconds: 300 + (index * 50)),
            curve: Curves.easeOutCubic,
            margin: const EdgeInsets.symmetric(vertical: 4),
            child: SlideTransition(
              position:
                  Tween<Offset>(
                    begin: Offset(message.role == MessageRole.user ? 1.0 : -1.0, 0.0),
                    end: Offset.zero,
                  ).animate(
                    CurvedAnimation(
                      parent: _fadeAnimationController,
                      curve: Interval(
                        (index * 0.1).clamp(0.0, 1.0),
                        ((index * 0.1) + 0.3).clamp(0.0, 1.0),
                        curve: Curves.easeOutCubic,
                      ),
                    ),
                  ),
              child: FadeTransition(
                opacity: CurvedAnimation(
                  parent: _fadeAnimationController,
                  curve: Interval(
                    (index * 0.1).clamp(0.0, 1.0),
                    ((index * 0.1) + 0.3).clamp(0.0, 1.0),
                    curve: Curves.easeOut,
                  ),
                ),
                child: MessageBubble(
                  messageId: messageId,
                  animated: true,
                  animationDuration: Duration(
                    milliseconds: 300 + (ref.watch(messageIdsProvider).indexOf(messageId) * 50.0).toInt(),
                  ),
                ),
              ),
            ),
          );
        },
      );
    }).toList();
  }

  // ignore: unused_element
  Widget _buildEditMessageBubble(MessageListState chatState, MessageListNotifier messageListNotifier) {
    // EditMessageBubble
    return Container(
      constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.78),
      margin: const EdgeInsets.only(right: 8),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Theme.of(context).brightness == Brightness.dark
            ? const Color(0xFF374151).withValues(alpha: 0.9)
            : Colors.white.withValues(alpha: 0.95),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppTheme.primaryBlue.withValues(alpha: 0.3)),
        boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.1), blurRadius: 16, offset: const Offset(0, 4))],
      ),
      child: Column(
        children: [
          TextFormField(
            initialValue: chatState.editText,
            onChanged: messageListNotifier.updateEditText,
            maxLines: 3,
            style: TextStyle(
              color: Theme.of(context).brightness == Brightness.dark ? Colors.white : const Color(0xFF111827),
              fontSize: 14.5,
            ),
            decoration: InputDecoration(
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(16),
                borderSide: BorderSide(color: Theme.of(context).dividerColor.withValues(alpha: 0.3)),
              ),
              enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(16),
                borderSide: BorderSide(color: Theme.of(context).dividerColor.withValues(alpha: 0.3)),
              ),
              focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(16),
                borderSide: BorderSide(color: AppTheme.primaryBlue, width: 2),
              ),
              hintText: 'Edit your message...',
              hintStyle: TextStyle(color: Theme.of(context).textTheme.bodySmall?.color?.withValues(alpha: 0.5)),
              contentPadding: const EdgeInsets.all(16),
            ),
            autofocus: true,
          ),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              TextButton(
                onPressed: messageListNotifier.cancelEditing,
                style: TextButton.styleFrom(foregroundColor: Theme.of(context).textTheme.bodyMedium?.color),
                child: const Text('Cancel'),
              ),
              const SizedBox(width: 12),
              ElevatedButton(
                onPressed: () {
                  messageListNotifier.sendMessage(
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
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                ),
                child: const Text('Save'),
              ),
            ],
          ),
        ],
      ),
    );
  }

  String _getStatusText(bool isRecording, bool isStreaming) {
    if (isRecording) {
      return '🔴 Recording... Tap mic to stop';
    } else if (isStreaming) {
      return 'Fitvise AI is thinking...';
    }
    return 'Shift+Enter for new line';
  }
}
