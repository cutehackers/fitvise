import 'dart:async';
import 'dart:collection';
import 'dart:convert';

import 'package:fitvise/providers/message_ids_provider.dart';
import 'package:fitvise/providers/message_provider.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;

import '../config/app_config.dart';
import '../models/chat_message.dart';
import '../models/chat_request.dart';
import '../models/chat_response.dart';
import '../models/message.dart';

/// Provider for accessing chat functionality
final messageListProvider = StateNotifierProvider<MessageListNotifier, MessageListState>((ref) {
  return MessageListNotifier(ref);
});

/// Chat state containing all chat-related data
class MessageListState {
  /// true if AI is currently streaming a response
  final bool isStreaming;

  /// true if recording audio input
  final bool isRecording;

  /// true if it is starting request / waiting for response
  final bool isLoading;

  final String? editingMessageId;
  final String editText;

  const MessageListState({
    this.isStreaming = false,
    this.isRecording = false,
    this.isLoading = false,
    this.editingMessageId,
    this.editText = '',
  });

  // List<Message> get messages => messageMap.values.toList();

  MessageListState copyWith({
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
    return MessageListState(
      isStreaming: isTyping ?? isStreaming,
      isRecording: isRecording ?? this.isRecording,
      isLoading: isLoading ?? this.isLoading,
      editingMessageId: editingMessageId ?? this.editingMessageId,
      editText: editText ?? this.editText,
    );
  }
}

/// Chat notifier provider using Riverpod for state management
class MessageListNotifier extends StateNotifier<MessageListState> {
  final Ref _ref;
  StreamSubscription? _streamSubscription;

  final LinkedHashMap<String, Message> messageMap = LinkedHashMap<String, Message>();
  final Map<String, StringBuffer> _buffers = {};

  MessageListNotifier(this._ref)
    : super(
        MessageListState(
          // messageMap: {
          //     '1': Message(
          //       id: '1',
          //       sender: 'ai',
          //       text:
          //           "Welcome to Fitvise! 💪 I'm your AI fitness assistant. I can help you with workout plans, nutrition advice, exercise techniques, and tracking your fitness progress. What would you like to work on today?",
          //       timestamp: DateTime.now(),
          //       type: 'text',
          //     ),
          //   },
        ),
      );

  @override
  void dispose() {
    _streamSubscription?.cancel();
    super.dispose();
  }

  /// Welcome prompts data - same as original implementation
  // List<WelcomePrompt> get welcomePrompts => [
  //   WelcomePrompt(icon: '🏋️', text: 'Create a personalized workout plan', category: 'Workout'),
  //   WelcomePrompt(icon: '🥗', text: 'Get nutrition and meal planning advice', category: 'Nutrition'),
  //   WelcomePrompt(icon: '📊', text: 'Track my fitness progress', category: 'Progress'),
  //   WelcomePrompt(icon: '💡', text: 'Learn proper exercise techniques', category: 'Education'),
  //   WelcomePrompt(icon: '🎯', text: 'Set and achieve fitness goals', category: 'Goals'),
  //   WelcomePrompt(icon: '🧘', text: 'Recovery and wellness tips', category: 'Wellness'),
  // ];

  Future<void> sendMessage(String sessionId, String text, {bool isEdit = false, String? editId}) async {
    final userQuery = text.trim();
    if (userQuery.isEmpty) return;

    try {
      // Process AI response - user message is safe in the list
      await _doSendMessage(sessionId, userQuery);
    } catch (error) {
      // Even on error, user message remains in the list
      _onMessageError(error.toString());
    } finally {
      state = state.copyWith(isLoading: false, isTyping: false);
    }
  }

  Future<void> _doSendMessage(String sessionId, String userQuery) async {
    // Add user message
    _addMessage(
      Message.user(id: DateTime.now().millisecondsSinceEpoch.toString(), text: userQuery, timestamp: DateTime.now()),
    );

    // Add ai message. begin streaming response with empty content.
    final messageId = DateTime.now().millisecondsSinceEpoch.toString();

    // prepare chat stream
    _buffers[messageId] = StringBuffer();
    _addMessage(
      Message.ai(
        id: messageId,
        text: '',
        timestamp: DateTime.now(),
        isStreaming: true, // Mark as streaming
      ),
    );

    state = state.copyWith(isLoading: true, isTyping: true, error: null, streamingMessageId: messageId);

    try {
      // Create an ai message stream
      final stream = _createChatStream(
        ChatRequest(
          model: 'llama3.2:3b',
          sessionId: sessionId,
          message: ChatMessage(role: 'user', content: userQuery),
          stream: true,
        ),
      );

      await for (final response in stream) {
        if (response.error != null) {
          throw Exception('Chat stream error: ${response.error}');
        }
        _appendToMessageContent(messageId, response);

        if (response.done) {
          break;
        }
      }

      _completeMessageContent(messageId);
    } catch (e, s) {
      _onChatStreamError(messageId, e, s);
      rethrow;
    } finally {
      _cleanUpChatStream(messageId);
    }
  }

  void _addMessage(Message message) {
    messageMap[message.id] = message;

    // Set the message content
    _ref.read(messageProvider(message.id).notifier).setMessage(message);

    // Add to message list
    _ref.read(messageIdsProvider.notifier).add(message.id);
  }

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
        .where((e) => e.trim().isNotEmpty)
        .map((e) => ChatResponse.fromJson(jsonDecode(e)));

    await for (final chat in stream) {
      yield chat;
    }
  }

  void _appendToMessageContent(String messageId, ChatResponse response) {
    final buffer = _buffers[messageId];
    if (buffer == null) {
      // Failed to append chunk to non-existent buffer, No buffer found for messageId
      return;
    }

    final chunk = response.message?.content ?? '';
    buffer.write(chunk);

    _ref.read(messageProvider(messageId).notifier).updateContent(buffer.toString());
  }

  void _completeMessageContent(String messageId) {
    final provider = _ref.read(messageProvider(messageId).notifier);

    final buffer = _buffers[messageId];
    if (buffer == null) {
      provider.updateContent(buffer.toString());
    }
  }

  void _onChatStreamError(String messageId, dynamic error, StackTrace stackTrace) {
    final provider = _ref.read(messageProvider(messageId).notifier);
    provider.update(
      text: '⚠️ $error',
      isStreaming: false,
      actions: [
        MessageAction(label: 'Try again', action: 'retry'),
        MessageAction(label: 'Contact support', action: 'support'),
      ],
    );

    _cleanUpChatStream(messageId);
  }

  /// Clean up streaming message on error - only removes empty AI messages
  void _cleanUpChatStream(String messageId) {
    _ref.read(messageProvider(messageId).notifier).setStreaming(false);
    _buffers.remove(messageId);
  }

  ///---------------

  /// Handle errors while preserving user messages
  void _onMessageError(String error) {}

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

  /// Clear chat - preserves welcome message
  void clearChat() {
    // final welcomeMessage = Message.ai(
    //   text:
    //       "Welcome to Fitvise! 💪 I'm your AI fitness assistant. I can help you with workout plans, nutrition advice, exercise techniques, and tracking your fitness progress. What would you like to work on today?",
    //   timestamp: DateTime.now(),
    // );

    _streamSubscription?.cancel();

    state = MessageListState();
  }
}
