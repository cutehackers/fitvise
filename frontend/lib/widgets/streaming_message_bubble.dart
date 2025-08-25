/// A message bubble widget that efficiently updates only for its specific message
// /// using the streaming mechanism instead of rebuilding the entire chat state
// class StreamingMessageBubble extends ConsumerWidget {
//   final Message message;
//   final VoidCallback? onEdit;
//   final VoidCallback? onDelete;

//   const StreamingMessageBubble({super.key, required this.message, this.onEdit, this.onDelete});

//   @override
//   Widget build(BuildContext context, WidgetRef ref) {
//     // For AI messages that are streaming, use the streaming provider
//     if (message.role == MessageRole.ai && message.isStreaming) {
//       return _buildStreamingBubble(context, ref);
//     }

//     // For completed messages, use the static content
//     return _buildStaticBubble(context);
//   }

//   /// Build a bubble that listens to streaming updates for this specific message
//   Widget _buildStreamingBubble(BuildContext context, WidgetRef ref) {
//     final streamAsyncValue = ref.watch(messageContentProvider(message.id));
//     return streamAsyncValue.when(
//       data: (streamingContent) => _buildBubbleContent(
//         context,
//         streamingContent.isNotEmpty ? streamingContent : message.text,
//         isStreaming: true,
//       ),
//       loading: () => _buildBubbleContent(context, message.text, isStreaming: true),
//       error: (error, stack) => _buildBubbleContent(
//         context,
//         message.text.isEmpty ? 'Failed to load message' : message.text,
//         isStreaming: false,
//       ),
//     );
//   }

//   /// Build a bubble with static content (for completed messages)
//   Widget _buildStaticBubble(BuildContext context) {
//     return _buildBubbleContent(context, message.text, isStreaming: false);
//   }

//   /// Build the actual bubble content with styling
//   Widget _buildBubbleContent(BuildContext context, String content, {required bool isStreaming}) {
//     final theme = Theme.of(context);
//     final isUser = message.role == MessageRole.user;

//     return Container(
//       margin: EdgeInsets.only(left: isUser ? 50 : 10, right: isUser ? 10 : 50, top: 5, bottom: 5),
//       child: Align(
//         alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
//         child: Container(
//           padding: const EdgeInsets.all(12),
//           decoration: BoxDecoration(
//             color: isUser ? theme.colorScheme.primary : theme.colorScheme.surface,
//             borderRadius: BorderRadius.circular(16),
//             border: !isUser ? Border.all(color: theme.colorScheme.outline.withOpacity(0.2)) : null,
//           ),
//           child: Column(
//             crossAxisAlignment: CrossAxisAlignment.start,
//             children: [
//               // Message content
//               Text(
//                 content,
//                 style: TextStyle(
//                   color: isUser ? theme.colorScheme.onPrimary : theme.colorScheme.onSurface,
//                   fontSize: 16,
//                 ),
//               ),

//               // Streaming indicator
//               if (isStreaming) ...[
//                 const SizedBox(height: 8),
//                 Row(
//                   mainAxisSize: MainAxisSize.min,
//                   children: [
//                     SizedBox(
//                       width: 12,
//                       height: 12,
//                       child: CircularProgressIndicator(
//                         strokeWidth: 2,
//                         valueColor: AlwaysStoppedAnimation<Color>(
//                           isUser
//                               ? theme.colorScheme.onPrimary.withOpacity(0.7)
//                               : theme.colorScheme.primary.withOpacity(0.7),
//                         ),
//                       ),
//                     ),
//                     const SizedBox(width: 8),
//                     Text(
//                       'AI is typing...',
//                       style: TextStyle(
//                         color: isUser
//                             ? theme.colorScheme.onPrimary.withOpacity(0.7)
//                             : theme.colorScheme.onSurface.withOpacity(0.7),
//                         fontSize: 12,
//                         fontStyle: FontStyle.italic,
//                       ),
//                     ),
//                   ],
//                 ),
//               ],

//               // Action buttons (edit/delete) for completed messages
//               if (!isStreaming && (onEdit != null || onDelete != null)) ...[
//                 const SizedBox(height: 8),
//                 Row(
//                   mainAxisSize: MainAxisSize.min,
//                   children: [
//                     if (onEdit != null)
//                       IconButton(
//                         icon: Icon(
//                           Icons.edit,
//                           size: 16,
//                           color: isUser
//                               ? theme.colorScheme.onPrimary.withOpacity(0.7)
//                               : theme.colorScheme.onSurface.withOpacity(0.7),
//                         ),
//                         onPressed: onEdit,
//                         constraints: const BoxConstraints(minWidth: 24, minHeight: 24),
//                         padding: EdgeInsets.zero,
//                       ),
//                     if (onDelete != null)
//                       IconButton(
//                         icon: Icon(
//                           Icons.delete,
//                           size: 16,
//                           color: isUser
//                               ? theme.colorScheme.onPrimary.withOpacity(0.7)
//                               : theme.colorScheme.onSurface.withOpacity(0.7),
//                         ),
//                         onPressed: onDelete,
//                         constraints: const BoxConstraints(minWidth: 24, minHeight: 24),
//                         padding: EdgeInsets.zero,
//                       ),
//                   ],
//                 ),
//               ],
//             ],
//           ),
//         ),
//       ),
//     );
//   }
// }
