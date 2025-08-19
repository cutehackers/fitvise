import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import '../models/message.dart';
import '../providers/chat_provider.dart';
import '../theme/app_theme.dart';

class MessageList extends StatelessWidget {
  final String sessionId;
  final List<Message> messages;

  const MessageList({super.key, required this.sessionId, required this.messages});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: messages.map((message) => MessageBubble(sessionId: sessionId, message: message)).toList(),
    );
  }
}

class MessageBubble extends StatelessWidget {
  final String sessionId;
  final Message message;

  const MessageBubble({super.key, required this.sessionId, required this.message});

  @override
  Widget build(BuildContext context) {
    final isUser = message.sender == 'user';
    final chatProvider = Provider.of<ChatProvider>(context, listen: false);

    return Container(
      margin: const EdgeInsets.symmetric(vertical: 4),
      child: Column(
        crossAxisAlignment: isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
        children: [
          // AI avatar and name
          if (!isUser)
            Padding(
              padding: const EdgeInsets.only(left: 8, bottom: 4),
              child: Row(
                children: [
                  Container(
                    width: 24,
                    height: 24,
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(
                        colors: [AppTheme.primaryBlue, AppTheme.secondaryPurple],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Icon(Icons.fitness_center, color: Colors.white, size: 12),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    'Fitvise AI',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: Theme.of(context).textTheme.bodySmall?.color?.withValues(alpha: 0.6),
                    ),
                  ),
                ],
              ),
            ),

          // Message content
          Consumer<ChatProvider>(
            builder: (context, chatProvider, child) {
              if (chatProvider.editingMessageId == message.id) {
                return _buildEditingInterface(context, chatProvider);
              }
              return _buildMessageContent(context, chatProvider);
            },
          ),

          // Action buttons for AI messages
          if (!isUser && message.actions != null && message.actions!.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(left: 32, top: 8),
              child: Wrap(
                spacing: 8,
                runSpacing: 8,
                children: message.actions!.map((action) {
                  return _ActionButton(
                    label: action.label,
                    onPressed: () => chatProvider.sendMessage(sessionId, action.label),
                  );
                }).toList(),
              ),
            ),

          // Timestamp
          Padding(
            padding: EdgeInsets.only(left: isUser ? 0 : 32, right: isUser ? 8 : 0, top: 4),
            child: Text(
              _formatTime(message.timestamp),
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: Theme.of(context).textTheme.bodySmall?.color?.withOpacity(0.5),
                fontSize: 11,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMessageContent(BuildContext context, ChatProvider chatProvider) {
    final isUser = message.sender == 'user';

    return Row(
      mainAxisAlignment: isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Flexible(
          child: Container(
            constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.75),
            child: Stack(
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  margin: EdgeInsets.only(left: isUser ? 0 : 8),
                  decoration: BoxDecoration(
                    color: isUser
                        ? AppTheme
                              .primaryBlue // bg-fitvise-primary
                        : (Theme.of(context).brightness == Brightness.dark
                              ? const Color(0xFF374151) // bg-gray-700 for dark mode
                              : Colors.white), // bg-white for light mode
                    borderRadius: BorderRadius.circular(20),
                    border: !isUser
                        ? Border.all(
                            color: Theme.of(context).brightness == Brightness.dark
                                ? const Color(0xFF4B5563) // border-gray-600
                                : const Color(0xFFE5E7EB), // border-gray-200
                          )
                        : null,
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        message.text,
                        style: TextStyle(
                          color: isUser
                              ? Colors.white
                              : (Theme.of(context).brightness == Brightness.dark
                                    ? Colors
                                          .white // text-white for dark mode AI messages
                                    : const Color(0xFF111827)), // text-gray-900 for light mode
                          fontSize: 14,
                          height: 1.4,
                        ),
                      ),
                      if (message.isEdited)
                        Padding(
                          padding: const EdgeInsets.only(top: 4),
                          child: Text(
                            'edited',
                            style: TextStyle(
                              color: isUser
                                  ? Colors.white.withOpacity(0.7)
                                  : Theme.of(context).textTheme.bodySmall?.color?.withOpacity(0.5),
                              fontSize: 11,
                              fontStyle: FontStyle.italic,
                            ),
                          ),
                        ),
                    ],
                  ),
                ),

                // Message actions (hover overlay)
                Positioned(
                  right: isUser ? -8 : null,
                  left: !isUser ? -8 : null,
                  top: 0,
                  bottom: 0,
                  child: MouseRegion(child: _buildMessageActions(context, chatProvider, isUser)),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildMessageActions(BuildContext context, ChatProvider chatProvider, bool isUser) {
    return SizedBox(
      width: 80,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // Copy button
          Container(
            margin: const EdgeInsets.symmetric(vertical: 2),
            child: Material(
              color: Theme.of(context).brightness == Brightness.dark
                  ? const Color(0xFF374151) // bg-gray-700 for dark
                  : Colors.white, // bg-white for light
              borderRadius: BorderRadius.circular(16),
              elevation: 2,
              child: InkWell(
                borderRadius: BorderRadius.circular(16),
                onTap: () => _copyToClipboard(context),
                child: Container(
                  padding: const EdgeInsets.all(6),
                  child: Icon(
                    Icons.copy,
                    size: 14,
                    color: Theme.of(context).brightness == Brightness.dark ? Colors.white : const Color(0xFF374151),
                  ),
                ),
              ),
            ),
          ),

          // Edit button for user messages
          if (isUser)
            Container(
              margin: const EdgeInsets.symmetric(vertical: 2),
              child: Material(
                color: Theme.of(context).brightness == Brightness.dark
                    ? const Color(0xFF374151) // bg-gray-700 for dark
                    : Colors.white, // bg-white for light
                borderRadius: BorderRadius.circular(16),
                elevation: 2,
                child: InkWell(
                  borderRadius: BorderRadius.circular(16),
                  onTap: () => chatProvider.startEditingMessage(message.id, message.text),
                  child: Container(
                    padding: const EdgeInsets.all(6),
                    child: Icon(
                      Icons.edit,
                      size: 14,
                      color: Theme.of(context).brightness == Brightness.dark ? Colors.white : const Color(0xFF374151),
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildEditingInterface(BuildContext context, ChatProvider chatProvider) {
    return Container(
      constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.75),
      margin: const EdgeInsets.only(right: 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Theme.of(context).cardColor,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Theme.of(context).dividerColor),
      ),
      child: Column(
        children: [
          TextFormField(
            initialValue: chatProvider.editText,
            onChanged: chatProvider.updateEditText,
            maxLines: 3,
            decoration: const InputDecoration(border: OutlineInputBorder(), hintText: 'Edit your message...'),
            autofocus: true,
          ),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              TextButton(onPressed: chatProvider.cancelEditing, child: const Text('Cancel')),
              const SizedBox(width: 8),
              ElevatedButton(
                onPressed: () {
                  chatProvider.sendMessage(sessionId, chatProvider.editText, isEdit: true, editId: message.id);
                },
                child: const Text('Save'),
              ),
            ],
          ),
        ],
      ),
    );
  }

  void _copyToClipboard(BuildContext context) {
    Clipboard.setData(ClipboardData(text: message.text));
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(const SnackBar(content: Text('Message copied to clipboard'), duration: Duration(seconds: 2)));
  }

  String _formatTime(DateTime timestamp) {
    final hour = timestamp.hour > 12 ? timestamp.hour - 12 : timestamp.hour;
    final period = timestamp.hour >= 12 ? 'PM' : 'AM';
    final minute = timestamp.minute.toString().padLeft(2, '0');
    return '$hour:$minute $period';
  }
}

class _ActionButton extends StatefulWidget {
  final String label;
  final VoidCallback onPressed;

  const _ActionButton({required this.label, required this.onPressed});

  @override
  State<_ActionButton> createState() => _ActionButtonState();
}

class _ActionButtonState extends State<_ActionButton> {
  bool _isHovered = false;

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      onEnter: (_) => setState(() => _isHovered = true),
      onExit: (_) => setState(() => _isHovered = false),
      child: GestureDetector(
        onTap: widget.onPressed,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
          decoration: BoxDecoration(
            color: _isHovered ? AppTheme.primaryBlue : Colors.transparent,
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color: _isHovered
                  ? AppTheme.primaryBlue
                  : (Theme.of(context).brightness == Brightness.dark
                        ? const Color(0xFF4B5563) // border-gray-600
                        : const Color(0xFFD1D5DB)), // border-gray-300
            ),
          ),
          child: Text(
            widget.label,
            style: TextStyle(
              fontSize: 14,
              color: _isHovered
                  ? Colors.white
                  : (Theme.of(context).brightness == Brightness.dark
                        ? const Color(0xFF9CA3AF) // text-gray-300
                        : const Color(0xFF4B5563)), // text-gray-600
            ),
          ),
        ),
      ),
    );
  }
}
