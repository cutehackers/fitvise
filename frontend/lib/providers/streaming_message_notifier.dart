import 'dart:async';

import 'package:flutter_riverpod/flutter_riverpod.dart';

/// Notifier for managing streaming message content independently from main chat state
class StreamingMessageNotifier extends StateNotifier<Map<String, String>> {
  StreamingMessageNotifier() : super({});

  final Map<String, StreamController<String>> _streamControllers = {};

  /// Create a stream for a specific message ID
  Stream<String> createMessageStream(String messageId) {
    if (_streamControllers.containsKey(messageId)) {
      return _streamControllers[messageId]!.stream;
    }

    final controller = StreamController<String>.broadcast();
    _streamControllers[messageId] = controller;
    return controller.stream;
  }

  /// Update streaming message content for a specific message ID
  void updateStreamingContent(String messageId, String content) {
    // Update the state for potential fallback access
    state = {...state, messageId: content};
    
    // Emit to the specific message stream if it exists
    final controller = _streamControllers[messageId];
    if (controller != null && !controller.isClosed) {
      controller.add(content);
    }
  }

  /// Finalize a streaming message and close its stream
  void finalizeMessage(String messageId, String finalContent) {
    // Update final state
    state = {...state, messageId: finalContent};
    
    // Close the stream controller
    final controller = _streamControllers[messageId];
    if (controller != null && !controller.isClosed) {
      controller.add(finalContent);
      controller.close();
      _streamControllers.remove(messageId);
    }
  }

  /// Clean up a message stream (e.g., on error)
  void cleanupMessage(String messageId) {
    // Remove from state
    final newState = Map<String, String>.from(state);
    newState.remove(messageId);
    state = newState;
    
    // Close and remove stream controller
    final controller = _streamControllers[messageId];
    if (controller != null && !controller.isClosed) {
      controller.close();
      _streamControllers.remove(messageId);
    }
  }

  /// Get current content for a message ID
  String? getMessageContent(String messageId) {
    return state[messageId];
  }

  @override
  void dispose() {
    // Close all stream controllers
    for (final controller in _streamControllers.values) {
      if (!controller.isClosed) {
        controller.close();
      }
    }
    _streamControllers.clear();
    super.dispose();
  }
}

/// Provider for streaming message content
final streamingMessageProvider = StateNotifierProvider<StreamingMessageNotifier, Map<String, String>>((ref) {
  return StreamingMessageNotifier();
});

/// Provider for a specific message stream
final messageStreamProvider = StreamProvider.family<String, String>((ref, messageId) {
  final streamingNotifier = ref.watch(streamingMessageProvider.notifier);
  return streamingNotifier.createMessageStream(messageId);
});