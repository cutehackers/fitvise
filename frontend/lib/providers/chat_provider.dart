import 'dart:async';
import 'dart:convert' show jsonDecode, jsonEncode, utf8, LineSplitter;

import 'package:dio/dio.dart';
import 'package:fitvise/config/app_config.dart';
import 'package:fitvise/models/chat_message.dart';
import 'package:fitvise/models/chat_request.dart';
import 'package:fitvise/models/chat_response.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import '../models/message.dart';

/// Enum of HTTP methods
enum HttpMethod { get, put, post, delete, options, head, patch, trace }

class ChatProvider extends ChangeNotifier {
  List<Message> _messages = [
    Message(
      id: '1',
      sender: 'ai',
      text:
          "Welcome to Fitvise! 💪 I'm your AI fitness assistant. I can help you with workout plans, nutrition advice, exercise techniques, and tracking your fitness progress. What would you like to work on today?",
      timestamp: DateTime.now(),
      type: 'text',
    ),
  ];

  bool _isTyping = false;
  bool _isRecording = false;
  bool _isLoading = false;
  bool _showWelcomePrompts = true;
  bool _showQuickReplies = true;
  String? _editingMessageId;
  String _editText = '';
  String? _streamingMessageId;
  String _streamingContent = '';

  // Initialize the API client
  ChatProvider();

  // Getters
  List<Message> get messages => _messages;
  bool get isTyping => _isTyping;
  bool get isRecording => _isRecording;
  bool get isLoading => _isLoading;
  bool get showWelcomePrompts => _showWelcomePrompts;
  bool get showQuickReplies => _showQuickReplies;
  String? get editingMessageId => _editingMessageId;
  String get editText => _editText;
  String? get streamingMessageId => _streamingMessageId;
  String get streamingContent => _streamingContent;

  // Welcome prompts data
  final List<WelcomePrompt> welcomePrompts = [
    WelcomePrompt(icon: '🏋️', text: 'Create a personalized workout plan', category: 'Workout'),
    WelcomePrompt(icon: '🥗', text: 'Get nutrition and meal planning advice', category: 'Nutrition'),
    WelcomePrompt(icon: '📊', text: 'Track my fitness progress', category: 'Progress'),
    WelcomePrompt(icon: '💡', text: 'Learn proper exercise techniques', category: 'Education'),
    WelcomePrompt(icon: '🎯', text: 'Set and achieve fitness goals', category: 'Goals'),
    WelcomePrompt(icon: '🧘', text: 'Recovery and wellness tips', category: 'Wellness'),
  ];

  // Quick replies data
  final List<String> quickReplies = [
    'Create a workout plan for me',
    'What are your fitness capabilities?',
    'Help me with nutrition advice',
    'Track my progress',
  ];

  // Send message
  Future<void> sendMessage(String sessionId, String text, {bool isEdit = false, String? editId}) async {
    if (text.trim().isEmpty) return;

    _showWelcomePrompts = false;
    _showQuickReplies = false;
    notifyListeners();

    if (isEdit && editId != null) {
      // Edit existing message
      final messageIndex = _messages.indexWhere((msg) => msg.id == editId);
      if (messageIndex != -1) {
        _messages[messageIndex] = _messages[messageIndex].copyWith(text: text, isEdited: true);
      }
      _editingMessageId = null;
      _editText = '';
      notifyListeners();
      return;
    }

    // Add new user message
    final userMessage = Message(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      sender: 'user',
      text: text,
      timestamp: DateTime.now(),
      type: 'text',
    );
    _messages.add(userMessage);
    notifyListeners();

    // Get AI response from backend
    _isTyping = true;
    _isLoading = true;
    notifyListeners();

    try {
      //await _prompt(text);
      await _chat(sessionId, text);

      // Note: The message is already added to _messages during streaming in _prompt method
      // So we don't need to add it again here
    } catch (error) {
      // Clear any streaming state on error
      _streamingMessageId = null;
      _streamingContent = '';

      // Add error message if API fails
      final errorMessage = _createErrorMessage(error);
      _messages.add(errorMessage);

      // Log error for debugging
      debugPrint('API Error: $error');
    }

    _isTyping = false;
    _isLoading = false;
    notifyListeners();
  }

  Future<void> _chat(String sessionId, String query) async {
    final request = ChatRequest(
      model: 'llama3.2:3b',
      sessionId: sessionId,
      message: ChatMessage(role: 'user', content: query),
      stream: true,
    );

    // Create streaming message
    final streamingMessageId = DateTime.now().millisecondsSinceEpoch.toString();
    _streamingMessageId = streamingMessageId;

    final streamingMessage = Message(
      id: streamingMessageId,
      sender: 'ai',
      text: '',
      timestamp: DateTime.now(),
      type: 'text',
    );
    _messages.add(streamingMessage);
    notifyListeners();

    try {
      final chatStream = _createChatStream(request);

      String chatContent = '';
      await for (final chunk in chatStream) {
        if (chunk.error != null) {
          throw Exception('Chat> stream error: ${chunk.error}');
        }

        final chatChunk = chunk.message?.content ?? '';
        if (chatChunk.isNotEmpty) {
          debugPrint('Chat> chunk received: $chatChunk');
          chatContent += chatChunk;
          _streamingContent = chatContent;

          // Update UI with each chunk
          _updateStreamingMessage(streamingMessageId, chatContent);
        }

        // Handle stream completion
        if (chunk.done) {
          debugPrint('Chat> stream completed');
          break;
        }
      }

      // finalize the message
      _finalizeStreamingMessage(streamingMessageId, chatContent);
    } catch (e) {
      _cleanupStreamingState(streamingMessageId);
      rethrow;
    }
  }

  /// 25.08.01
  /// This method fetches the chat response from the backend using a streaming approach.
  /// It sends a POST request with the chat request data and processes the response stream.
  /// It is not possible to get streaming response with Dio or Rerofit at the moment.
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

  // Update streaming message UI (similar to LangChain.dart's chunk processing)
  void _updateStreamingMessage(String messageId, String content) {
    final messageIndex = _messages.indexWhere((msg) => msg.id == messageId);
    if (messageIndex != -1) {
      _messages[messageIndex] = _messages[messageIndex].copyWith(text: content);
      notifyListeners(); // Trigger UI update
    }
  }

  // Finalize streaming message (LangChain.dart completion pattern)
  Message _finalizeStreamingMessage(String messageId, String content) {
    _streamingMessageId = null;
    _streamingContent = '';

    final messageIndex = _messages.indexWhere((msg) => msg.id == messageId);
    if (messageIndex != -1) {
      final finalText = content.isEmpty
          ? 'I apologize, but I didn\'t receive a complete response. Please try again.'
          : content;

      _messages[messageIndex] = _messages[messageIndex].copyWith(text: finalText, actions: <MessageAction>[]);
      notifyListeners();

      return _messages[messageIndex];
    }

    // Fallback message
    return Message(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      sender: 'ai',
      text: content.isEmpty ? 'I apologize, but I didn\'t receive a complete response. Please try again.' : content,
      timestamp: DateTime.now(),
      type: 'text',
      actions: <MessageAction>[],
    );
  }

  // Cleanup streaming state on error (LangChain.dart error handling)
  void _cleanupStreamingState(String messageId) {
    _streamingMessageId = null;
    _streamingContent = '';

    // Only remove the specific streaming message if it exists and is empty
    final messageIndex = _messages.indexWhere((msg) => msg.id == messageId);
    if (messageIndex != -1 && _messages[messageIndex].sender == 'ai' && _messages[messageIndex].text.isEmpty) {
      _messages.removeAt(messageIndex);
      notifyListeners();
    }
  }

  Message _createErrorMessage(dynamic error) {
    String errorText;

    if (error is DioException) {
      switch (error.type) {
        case DioExceptionType.connectionTimeout:
        case DioExceptionType.sendTimeout:
        case DioExceptionType.receiveTimeout:
          errorText = "⏰ Request timed out. Please check your internet connection and try again.";
          break;
        case DioExceptionType.connectionError:
          errorText = "🌐 Connection error. Please check your internet connection and try again.";
          break;
        case DioExceptionType.badResponse:
          switch (error.response?.statusCode) {
            case 401:
              errorText = "🔐 Authentication failed. Please check your credentials.";
              break;
            case 404:
              errorText = "❓ Service not found. The fitness API might be temporarily unavailable.";
              break;
            case 429:
              errorText = "⚡ Too many requests. Please wait a moment before trying again.";
              break;
            case 500:
              errorText = "🔧 Server error. Our fitness AI is having technical difficulties.";
              break;
            case 503:
              errorText = "🚧 Service temporarily unavailable. Please try again in a few minutes.";
              break;
            default:
              errorText = "⚠️ Something went wrong (Error ${error.response?.statusCode}). Please try again.";
          }
          break;
        case DioExceptionType.cancel:
          errorText = "❌ Request was cancelled. Please try again.";
          break;
        default:
          errorText = "🤖 I'm having trouble connecting to my fitness knowledge base. Please try again!";
      }
    } else {
      errorText = "💪 Sorry, I encountered an unexpected issue. Please try again in a moment!";
    }

    return Message(
      id: (DateTime.now().millisecondsSinceEpoch + 1).toString(),
      sender: 'ai',
      text: errorText,
      timestamp: DateTime.now(),
      type: 'text',
      actions: [
        MessageAction(label: 'Try again', action: 'retry'),
        MessageAction(label: 'Contact support', action: 'support'),
      ],
    );
  }

  // Start editing message
  void startEditingMessage(String messageId, String currentText) {
    _editingMessageId = messageId;
    _editText = currentText;
    notifyListeners();
  }

  // Cancel editing
  void cancelEditing() {
    _editingMessageId = null;
    _editText = '';
    notifyListeners();
  }

  // Update edit text
  void updateEditText(String text) {
    _editText = text;
    notifyListeners();
  }

  // Toggle recording
  void toggleRecording() {
    _isRecording = !_isRecording;
    notifyListeners();
  }

  // Send quick reply
  Future<void> sendQuickReply(String sessionId, String reply) async {
    await sendMessage(sessionId, reply);
  }

  // Send welcome prompt
  Future<void> sendWelcomePrompt(String sessionId, WelcomePrompt prompt) async {
    await sendMessage(sessionId, prompt.text);
  }

  // Copy message to clipboard
  Future<void> copyMessage(String text) async {
    // In a real app, you'd use a clipboard package
    // For now, this is a placeholder
  }

  // Clear chat
  void clearChat() {
    _messages = [_messages.first]; // Keep welcome message
    _showWelcomePrompts = true;
    _showQuickReplies = true;
    _editingMessageId = null;
    _editText = '';
    notifyListeners();
  }
}
