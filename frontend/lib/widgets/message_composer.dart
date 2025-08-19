import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/chat_provider.dart';

/// A comprehensive message composition widget for fitness chat interfaces.
///
/// The [MessageComposer] provides a complete input interface for creating and
/// sending messages in fitness applications. It includes text input, voice
/// recording, file attachments, and real-time status feedback.
///
/// {@tool snippet}
/// Basic usage with a text controller:
///
/// ```dart
/// final TextEditingController controller = TextEditingController();
///
/// MessageComposer(
///   textController: controller,
/// )
/// ```
/// {@end-tool}
///
/// The widget integrates with [ChatProvider] for state management and provides:
/// * Multi-line text input with character count (2000 max)
/// * Voice recording with visual feedback
/// * File attachment options (images, documents, files)
/// * Send button with loading states
/// * Real-time typing indicators
/// * Keyboard shortcuts (Shift+Enter for new line)
///
/// The [textController] parameter is required and manages the input text state.
/// The widget automatically handles focus management, validation, and cleanup.
///
/// See also:
///
///  * [ChatProvider], which manages chat state and message sending
///  * [TextField], the underlying input widget
///  * [Consumer], used for provider integration
class MessageComposer extends StatefulWidget {
  /// The session ID for the chat.
  ///
  /// This ID is used to identify the chat session and is used to store the chat history.
  final String sessionId;

  /// The text editing controller for the message input field.
  ///
  /// This controller manages the text content of the input field and
  /// allows external access to the current text. The widget listens
  /// to changes on this controller to update the send button state.
  ///
  /// Must not be null.
  final TextEditingController textController;

  /// Creates a message composer widget.
  ///
  /// The [textController] parameter is required and must not be null.
  /// It manages the text input state and allows external access to the
  /// current input text.
  ///
  /// The widget will automatically add and remove listeners to the
  /// [textController] for internal state management.
  const MessageComposer({super.key, required this.sessionId, required this.textController});

  @override
  State<MessageComposer> createState() => _MessageComposerState();
}

/// State class for [MessageComposer].
///
/// Manages the widget's internal state including:
/// * Text controller listener for send button state updates
/// * Focus node for keyboard focus management
/// * Message sending and file upload handling
/// * Status text generation based on chat provider state
class _MessageComposerState extends State<MessageComposer> {
  final FocusNode _focusNode = FocusNode();

  @override
  void initState() {
    super.initState();
    widget.textController.addListener(_onTextChanged);
  }

  @override
  void dispose() {
    widget.textController.removeListener(_onTextChanged);
    _focusNode.dispose();
    super.dispose();
  }

  void _onTextChanged() {
    setState(() {
      // Rebuild to update send button state
    });
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<ChatProvider>(
      builder: (context, chatProvider, child) {
        return Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Theme.of(context).scaffoldBackgroundColor,
            border: Border(top: BorderSide(color: Theme.of(context).dividerColor)),
          ),
          child: Column(
            children: [
              // Attachment options
              Row(
                children: [
                  _AttachmentButton(
                    icon: Icons.image,
                    tooltip: 'Upload workout photo',
                    onPressed: () => _handleFileUpload('image'),
                  ),
                  const SizedBox(width: 8),
                  _AttachmentButton(
                    icon: Icons.attach_file,
                    tooltip: 'Upload file',
                    onPressed: () => _handleFileUpload('file'),
                  ),
                  const SizedBox(width: 8),
                  _AttachmentButton(
                    icon: Icons.description,
                    tooltip: 'Upload fitness plan',
                    onPressed: () => _handleFileUpload('document'),
                  ),
                ],
              ),
              const SizedBox(height: 12),

              // Input field
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Theme.of(context).brightness == Brightness.dark
                      ? const Color(0xFF374151) // bg-gray-700
                      : Colors.white, // bg-white
                  borderRadius: BorderRadius.circular(24),
                  border: Border.all(
                    color: Theme.of(context).brightness == Brightness.dark
                        ? const Color(0xFF4B5563) // border-gray-600
                        : const Color(0xFFD1D5DB), // border-gray-300
                    width: 2,
                  ),
                ),
                child: Row(
                  children: [
                    // Text input
                    Expanded(
                      child: TextField(
                        controller: widget.textController,
                        focusNode: _focusNode,
                        maxLines: null,
                        textCapitalization: TextCapitalization.sentences,
                        textInputAction: TextInputAction.send,
                        decoration: InputDecoration(
                          hintText: 'Ask about workouts, nutrition, or fitness goals... (Shift+Enter for new line)',
                          hintStyle: TextStyle(
                            color: Theme.of(context).brightness == Brightness.dark
                                ? const Color(0xFF9CA3AF) // placeholder-gray-400
                                : const Color(0xFF6B7280), // placeholder-gray-500
                          ),
                          border: InputBorder.none,
                          contentPadding: EdgeInsets.zero,
                        ),
                        style: TextStyle(
                          color: Theme.of(context).brightness == Brightness.dark
                              ? Colors
                                    .white // text-white
                              : const Color(0xFF111827), // text-gray-900
                        ),
                        onSubmitted: (text) {
                          if (text.trim().isNotEmpty) {
                            _sendMessage(widget.sessionId, chatProvider);
                          }
                        },
                      ),
                    ),
                    const SizedBox(width: 8),

                    // Voice recording button
                    IconButton(
                      onPressed: chatProvider.toggleRecording,
                      icon: Icon(
                        chatProvider.isRecording ? Icons.mic_off : Icons.mic,
                        color: chatProvider.isRecording ? Colors.red : Theme.of(context).iconTheme.color,
                      ),
                      tooltip: chatProvider.isRecording ? 'Stop recording' : 'Start voice input',
                    ),

                    // Send button
                    IconButton(
                      onPressed: widget.textController.text.trim().isNotEmpty && !chatProvider.isLoading
                          ? () => _sendMessage(widget.sessionId, chatProvider)
                          : null,
                      icon: chatProvider.isLoading
                          ? SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                valueColor: AlwaysStoppedAnimation<Color>(Theme.of(context).primaryColor),
                              ),
                            )
                          : Icon(
                              Icons.send,
                              color: widget.textController.text.trim().isNotEmpty
                                  ? Theme.of(context).iconTheme.color
                                  : Theme.of(context).disabledColor,
                            ),
                      tooltip: 'Send message',
                    ),
                  ],
                ),
              ),

              // Status and character count
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    // Status text
                    Expanded(
                      child: Text(
                        _getStatusText(chatProvider),
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: Theme.of(context).textTheme.bodySmall?.color?.withOpacity(0.6),
                        ),
                      ),
                    ),

                    // Character count
                    Text(
                      '${widget.textController.text.length}/2000',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: Theme.of(context).textTheme.bodySmall?.color?.withOpacity(0.6),
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

  void _sendMessage(String sessionId, ChatProvider chatProvider) {
    final text = widget.textController.text.trim();
    if (text.isNotEmpty) {
      chatProvider.sendMessage(sessionId, text);
      widget.textController.clear();
      _focusNode.requestFocus();
    }
  }

  void _handleFileUpload(String type) {
    // TODO: Implement file upload
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text('File upload ($type) coming soon!'), duration: const Duration(seconds: 2)));
  }

  String _getStatusText(ChatProvider chatProvider) {
    if (chatProvider.isRecording) {
      return '🔴 Recording... Tap mic to stop';
    } else if (chatProvider.isTyping) {
      return 'Fitvise AI is thinking...';
    }
    return '';
  }
}

class _AttachmentButton extends StatelessWidget {
  final IconData icon;
  final String tooltip;
  final VoidCallback onPressed;

  const _AttachmentButton({required this.icon, required this.tooltip, required this.onPressed});

  @override
  Widget build(BuildContext context) {
    return IconButton(
      onPressed: onPressed,
      icon: Icon(icon, size: 18),
      tooltip: tooltip,
      constraints: const BoxConstraints(minWidth: 32, minHeight: 32),
      style: IconButton.styleFrom(
        backgroundColor: Colors.transparent,
        foregroundColor: Theme.of(context).iconTheme.color?.withValues(alpha: .7),
      ),
    );
  }
}
