# Streaming Message Architecture

## Overview

The `messageStreamProvider` is designed to optimize chat message updates by providing individual streams for each message ID. This eliminates the need to rebuild the entire chat widget tree when only one message is being updated during streaming.

## Key Components

### 1. StreamingMessageNotifier
- **Location**: `lib/providers/streaming_message_notifier.dart`
- **Purpose**: Manages individual streams for each message ID
- **Benefits**: Reactive updates, memory efficiency, automatic cleanup

### 2. Message Stream Provider
```dart
final messageStreamProvider = StreamProvider.family<String, String>((ref, messageId) {
  final streamingNotifier = ref.watch(streamingMessageProvider.notifier);
  return streamingNotifier.createMessageStream(messageId);
});
```

### 3. Usage in Message Bubbles

#### Traditional Approach (Inefficient)
```dart
// ❌ Old way - rebuilds entire chat state
class MessageBubble extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final chatState = ref.watch(chatNotifierProvider);
    final streamingContent = chatState.streamingContent;
    // Widget rebuilds every time any message changes
  }
}
```

#### Optimized Streaming Approach (Efficient)
```dart
// ✅ New way - only rebuilds specific message
class StreamingMessageBubble extends ConsumerWidget {
  final Message message;
  
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    if (message.isStreaming) {
      // Subscribe to this specific message's stream
      final streamAsyncValue = ref.watch(messageStreamProvider(message.id));
      return streamAsyncValue.when(
        data: (content) => _buildBubble(content),
        loading: () => _buildBubble(message.text),
        error: (_, __) => _buildBubble(message.text),
      );
    }
    return _buildBubble(message.text); // Static content
  }
}
```

## How It Works

### 1. Message Creation
When an AI response starts streaming:
```dart
final streamingMessage = Message(
  id: streamingMessageId,
  sender: 'ai',
  text: '',
  timestamp: DateTime.now(),
  isStreaming: true, // Triggers streaming UI optimization
);
```

### 2. Stream Updates
During NDJSON streaming:
```dart
void _updateStreamingMessage(String messageId, String content) {
  // Update the specific message stream (efficient)
  _streamingNotifier.updateStreamingContent(messageId, content);
  
  // Update state for fallback compatibility
  state = state.copyWith(streamingContent: content);
}
```

### 3. Stream Completion
When streaming finishes:
```dart
void _finalizeStreamingMessage(String messageId, String content) {
  // Update message to completed state
  updatedMessage = message.copyWith(
    text: finalText,
    isStreaming: false, // Stops streaming UI
  );
  
  // Close the stream and cleanup
  _streamingNotifier.finalizeMessage(messageId, finalText);
}
```

## Performance Benefits

### Memory Efficiency
- **Individual Streams**: Each message has its own stream controller
- **Automatic Cleanup**: Streams are closed when messages are finalized
- **Broadcast Streams**: Multiple widgets can listen to the same message stream

### UI Performance
- **Targeted Updates**: Only the specific message bubble rebuilds
- **Reduced Redraws**: Chat list doesn't rebuild for streaming updates
- **Responsive UI**: Smooth streaming without UI lag

### NDJSON Optimization
- **Streaming Protocol**: Designed specifically for `application/x-ndjson` responses
- **Real-time Updates**: Immediate display of streaming content
- **Error Handling**: Graceful fallback on stream failures

## Implementation Example

### In Your Message List Widget
```dart
class MessageListWidget extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final chatState = ref.watch(chatNotifierProvider);
    
    return ListView.builder(
      itemCount: chatState.messages.length,
      itemBuilder: (context, index) {
        final message = chatState.messages[index];
        
        // Use StreamingMessageBubble for AI messages
        if (message.sender == 'ai') {
          return StreamingMessageBubble(
            message: message,
            onEdit: () => _editMessage(message.id),
            onDelete: () => _deleteMessage(message.id),
          );
        }
        
        // Use regular bubble for user messages
        return RegularMessageBubble(message: message);
      },
    );
  }
}
```

### Benefits Over State-Based Updates

| Aspect | State-Based (Old) | Streaming-Based (New) |
|--------|------------------|----------------------|
| **Widget Rebuilds** | Entire chat list | Single message bubble |
| **Memory Usage** | High (full state) | Low (targeted streams) |
| **NDJSON Efficiency** | Poor (frequent rebuilds) | Excellent (reactive streams) |
| **UI Responsiveness** | Laggy during streaming | Smooth real-time updates |
| **Error Recovery** | Complex state management | Isolated stream handling |

## Migration Guide

### Step 1: Update Message Model
Add `isStreaming` property to your Message class (already done).

### Step 2: Replace Message Bubbles
Replace `MessageBubble` with `StreamingMessageBubble` for AI messages.

### Step 3: Use Stream Provider
Subscribe to `messageStreamProvider(messageId)` for streaming content.

### Step 4: Test Performance
Verify that only individual message bubbles rebuild during streaming.

## Debugging

### Check Stream Status
```dart
// In development, check if streams are properly created/closed
final streamingNotifier = ref.read(streamingMessageProvider.notifier);
final content = streamingNotifier.getMessageContent(messageId);
print('Current streaming content for $messageId: $content');
```

### Monitor Performance
```dart
// Track widget rebuilds during streaming
class StreamingMessageBubble extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    print('StreamingMessageBubble rebuilding for message: ${message.id}');
    // ... rest of implementation
  }
}
```

This architecture ensures efficient, responsive chat UI that scales well with message volume and provides smooth streaming experiences for users.