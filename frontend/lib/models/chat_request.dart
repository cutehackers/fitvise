// ignore_for_file: invalid_annotation_target

import 'package:freezed_annotation/freezed_annotation.dart';

import 'chat_message.dart';

part 'chat_request.freezed.dart';
part 'chat_request.g.dart';

/// Request model for chat completion (replaces PromptRequest)
@freezed
abstract class ChatRequest with _$ChatRequest {
  const factory ChatRequest({
    required String model,
    @JsonKey(name: 'session_id') required String sessionId,
    required ChatMessage message,
    List<Map<String, dynamic>>? tools,
    @Default(false) bool think,
    String? format,
    Map<String, dynamic>? options,
    @Default(true) bool stream,
    @JsonKey(name: 'keep_alive') String? keepAlive,
  }) = _ChatRequest;

  factory ChatRequest.fromJson(Map<String, dynamic> json) =>
      _$ChatRequestFromJson(json);
}
