import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;

import '../config/app_config.dart';
import '../models/chat_message.dart';
import '../models/chat_request.dart';
import '../models/chat_response.dart';
import '../models/message.dart';
import 'streaming_message_notifier.dart';

/// Chat state containing all chat-related data
class ChatState {
  final List<Message> messages;
  final bool isTyping;
  final bool isRecording;
  final bool isLoading;
  final bool showWelcomePrompts;
  final bool showQuickReplies;
  final String? editingMessageId;
  final String editText;
  final String? streamingMessageId;
  final String streamingContent;
  final String? error;

  const ChatState({
    this.messages = const [],
    this.isTyping = false,
    this.isRecording = false,
    this.isLoading = false,
    this.showWelcomePrompts = true,
    this.showQuickReplies = true,
    this.editingMessageId,
    this.editText = '',
    this.streamingMessageId,
    this.streamingContent = '',
    this.error,
  });

  ChatState copyWith({
    List<Message>? messages,
    bool? isTyping,
    bool? isRecording,
    bool? isLoading,
    bool? showWelcomePrompts,
    bool? showQuickReplies,
    String? editingMessageId,
    String? editText,
    String? streamingMessageId,
    String? streamingContent,
    String? error,
  }) {
    return ChatState(
      messages: messages ?? this.messages,
      isTyping: isTyping ?? this.isTyping,
      isRecording: isRecording ?? this.isRecording,
      isLoading: isLoading ?? this.isLoading,
      showWelcomePrompts: showWelcomePrompts ?? this.showWelcomePrompts,
      showQuickReplies: showQuickReplies ?? this.showQuickReplies,
      editingMessageId: editingMessageId ?? this.editingMessageId,
      editText: editText ?? this.editText,
      streamingMessageId: streamingMessageId ?? this.streamingMessageId,
      streamingContent: streamingContent ?? this.streamingContent,
      error: error ?? this.error,
    );
  }
}

/// Chat notifier provider using Riverpod for state management
class ChatNotifier extends StateNotifier<ChatState> {
  final Ref _ref;
  StreamSubscription? _streamSubscription;

  ChatNotifier(this._ref)
    : super(
        ChatState(
          messages: [
            Message(
              id: '1',
              sender: 'ai',
              text:
                  "Welcome to Fitvise! 💪 I'm your AI fitness assistant. I can help you with workout plans, nutrition advice, exercise techniques, and tracking your fitness progress. What would you like to work on today?",
              timestamp: DateTime.now(),
              type: 'text',
            ),
          ],
        ),
      );

  /// Get the streaming message notifier
  StreamingMessageNotifier get _streamingNotifier => _ref.read(streamingMessageProvider.notifier);

  @override
  void dispose() {
    _streamSubscription?.cancel();
    super.dispose();
  }

  /// Welcome prompts data - same as original implementation
  List<WelcomePrompt> get welcomePrompts => [
    WelcomePrompt(icon: '🏋️', text: 'Create a personalized workout plan', category: 'Workout'),
    WelcomePrompt(icon: '🥗', text: 'Get nutrition and meal planning advice', category: 'Nutrition'),
    WelcomePrompt(icon: '📊', text: 'Track my fitness progress', category: 'Progress'),
    WelcomePrompt(icon: '💡', text: 'Learn proper exercise techniques', category: 'Education'),
    WelcomePrompt(icon: '🎯', text: 'Set and achieve fitness goals', category: 'Goals'),
    WelcomePrompt(icon: '🧘', text: 'Recovery and wellness tips', category: 'Wellness'),
  ];

  /// Quick replies data - same as original implementation
  List<String> get quickReplies => [
    'Create a workout plan for me',
    'What are your fitness capabilities?',
    'Help me with nutrition advice',
    'Track my progress',
  ];

  /// Send message - FIXED VERSION that never removes user messages
  Future<void> sendMessage(String sessionId, String text, {bool isEdit = false, String? editId}) async {
    if (text.trim().isEmpty) return;

    final currentState = state;

    // Hide welcome prompts and quick replies
    state = currentState.copyWith(showWelcomePrompts: false, showQuickReplies: false);

    if (isEdit && editId != null) {
      // Handle message editing
      _editMessage(editId, text);
      return;
    }

    // Create and add user message - THIS MESSAGE IS NEVER REMOVED
    final userMessage = Message(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      sender: 'user',
      text: text,
      timestamp: DateTime.now(),
      type: 'text',
    );

    // Add user message to the list - this is permanent and immutable
    final updatedMessages = [...currentState.messages, userMessage];
    state = currentState.copyWith(messages: updatedMessages, isLoading: true, isTyping: true, error: null);

    try {
      // Process AI response - user message is safe in the list
      await _processAIResponse(sessionId, text);
    } catch (error) {
      // Even on error, user message remains in the list
      _handleError(error.toString());
    } finally {
      state = state.copyWith(isLoading: false, isTyping: false);
    }
  }

  /// Process AI response without affecting user messages
  Future<void> _processAIResponse(String sessionId, String query) async {
    final request = ChatRequest(
      model: 'llama3.2:3b',
      sessionId: sessionId,
      message: ChatMessage(role: 'user', content: query),
      stream: true,
    );

    // Create streaming AI message placeholder
    final streamingMessageId = DateTime.now().millisecondsSinceEpoch.toString();
    final streamingMessage = Message(
      id: streamingMessageId,
      sender: 'ai',
      text: '',
      timestamp: DateTime.now(),
      type: 'text',
      isStreaming: true, // Mark as streaming for UI optimization
    );

    // Add AI message placeholder to the list
    final currentMessages = state.messages;
    state = state.copyWith(messages: [...currentMessages, streamingMessage], streamingMessageId: streamingMessageId);

    try {
      final chatStream = _createChatStream(request);
      String chatContent = '';

      await for (final chunk in chatStream) {
        if (chunk.error != null) {
          throw Exception('Chat stream error: ${chunk.error}');
        }

        final chatChunk = chunk.message?.content ?? '';
        if (chatChunk.isNotEmpty) {
          chatContent += chatChunk;
          _updateStreamingMessage(streamingMessageId, chatContent);
        }

        if (chunk.done) {
          break;
        }
      }

      // Finalize the AI message
      _finalizeStreamingMessage(streamingMessageId, chatContent);
    } catch (e) {
      // On error, remove only the empty AI message, never user messages
      _cleanupStreamingMessage(streamingMessageId);
      rethrow;
    }
  }

  /// Create chat stream - same as original but with better error handling
  Stream<ChatResponse> _createChatStream(ChatRequest request) async* {
    final action = http.Request('POST', Uri.parse('${AppConfig.apiBaseUrl}/fitvise/chat'))
      ..headers.addAll({'Content-Type': 'application/json', 'Accept': 'application/x-ndjson'})
      ..body = jsonEncode(request.toJson());

    debugPrint('Chat> sending request: ${jsonEncode(request.toJson())}');

    final response = await action.send();
    if (response.statusCode != 200) {
      yield ChatResponse(
        model: request.model,
        createdAt: DateTime.now().toIso8601String(),
        done: true,
        success: false,
        error: 'HTTP ${response.statusCode}: ${response.reasonPhrase}',
      );
      return;
    }

    final stream = response.stream
        .transform(utf8.decoder)
        .transform(const LineSplitter())
        .where((e) => e.isNotEmpty)
        .map((e) => ChatResponse.fromJson(jsonDecode(e)));

    await for (final chat in stream) {
      yield chat;
    }
  }

  /// Update streaming message content using reactive stream - more efficient for NDJSON
  void _updateStreamingMessage(String messageId, String content) {
    // Use the streaming notifier to update only the specific message without rebuilding entire state
    _streamingNotifier.updateStreamingContent(messageId, content);
    
    // Update streaming content in state for fallback/compatibility
    state = state.copyWith(streamingContent: content);
  }

  /// Finalize streaming message - only affects AI messages
  void _finalizeStreamingMessage(String messageId, String content) {
    final currentMessages = state.messages;
    final messageIndex = currentMessages.indexWhere((msg) => msg.id == messageId);

    if (messageIndex != -1 && currentMessages[messageIndex].sender == 'ai') {
      final finalText = content.isEmpty
          ? 'I apologize, but I didn\'t receive a complete response. Please try again.'
          : content;

      final updatedMessages = [...currentMessages];
      updatedMessages[messageIndex] = currentMessages[messageIndex].copyWith(
        text: finalText,
        actions: <MessageAction>[],
        isStreaming: false, // Mark as completed streaming
      );

      // Finalize the streaming message and clean up its stream
      _streamingNotifier.finalizeMessage(messageId, finalText);

      state = state.copyWith(messages: updatedMessages, streamingMessageId: null, streamingContent: '');
    }
  }

  /// Clean up streaming message on error - only removes empty AI messages
  void _cleanupStreamingMessage(String messageId) {
    final currentMessages = state.messages;
    final messageIndex = currentMessages.indexWhere((msg) => msg.id == messageId);

    // Only remove empty AI messages, never user messages
    if (messageIndex != -1 &&
        currentMessages[messageIndex].sender == 'ai' &&
        currentMessages[messageIndex].text.isEmpty) {
      final updatedMessages = [...currentMessages];
      updatedMessages.removeAt(messageIndex);

      // Clean up the streaming message and close its stream
      _streamingNotifier.cleanupMessage(messageId);

      state = state.copyWith(messages: updatedMessages, streamingMessageId: null, streamingContent: '');
    }
  }

  /// Edit message implementation
  void _editMessage(String editId, String newText) {
    final currentMessages = state.messages;
    final messageIndex = currentMessages.indexWhere((msg) => msg.id == editId);

    if (messageIndex != -1) {
      final updatedMessages = [...currentMessages];
      updatedMessages[messageIndex] = currentMessages[messageIndex].copyWith(text: newText, isEdited: true);

      state = state.copyWith(messages: updatedMessages, editingMessageId: null, editText: '');
    }
  }

  /// Handle errors while preserving user messages
  void _handleError(String error) {
    // Clean up any incomplete streaming message
    final streamingId = state.streamingMessageId;
    if (streamingId != null) {
      _cleanupStreamingMessage(streamingId);
    }

    // Add error message without affecting user messages
    final errorMessage = Message(
      id: (DateTime.now().millisecondsSinceEpoch + 1).toString(),
      sender: 'ai',
      text: '⚠️ $error',
      timestamp: DateTime.now(),
      type: 'text',
      actions: [
        MessageAction(label: 'Try again', action: 'retry'),
        MessageAction(label: 'Contact support', action: 'support'),
      ],
    );

    final currentMessages = state.messages;
    state = state.copyWith(messages: [...currentMessages, errorMessage], error: error);
  }

  /// Start editing message
  void startEditingMessage(String messageId, String currentText) {
    state = state.copyWith(editingMessageId: messageId, editText: currentText);
  }

  /// Cancel editing
  void cancelEditing() {
    state = state.copyWith(editingMessageId: null, editText: '');
  }

  /// Update edit text
  void updateEditText(String text) {
    state = state.copyWith(editText: text);
  }

  /// Toggle recording
  void toggleRecording() {
    state = state.copyWith(isRecording: !state.isRecording);
  }

  /// Send quick reply
  Future<void> sendQuickReply(String sessionId, String reply) async {
    await sendMessage(sessionId, reply);
  }

  /// Send welcome prompt
  Future<void> sendWelcomePrompt(String sessionId, WelcomePrompt prompt) async {
    await sendMessage(sessionId, prompt.text);
  }

  /// Clear chat - preserves welcome message
  void clearChat() {
    final welcomeMessage = Message(
      id: '1',
      sender: 'ai',
      text:
          "Welcome to Fitvise! 💪 I'm your AI fitness assistant. I can help you with workout plans, nutrition advice, exercise techniques, and tracking your fitness progress. What would you like to work on today?",
      timestamp: DateTime.now(),
      type: 'text',
    );

    _streamSubscription?.cancel();

    state = ChatState(messages: [welcomeMessage], showWelcomePrompts: true, showQuickReplies: true);
  }
}

/// Provider for accessing chat functionality
final chatNotifierProvider = StateNotifierProvider<ChatNotifier, ChatState>((ref) {
  return ChatNotifier(ref);
});
