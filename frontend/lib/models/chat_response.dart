// ignore_for_file: invalid_annotation_target

import 'package:freezed_annotation/freezed_annotation.dart';

import 'chat_message.dart';

part 'chat_response.freezed.dart';
part 'chat_response.g.dart';

/// Response model for chat completion (replaces PromptResponse)
@freezed
abstract class ChatResponse with _$ChatResponse {
  const factory ChatResponse({
    required String model,
    @JsonKey(name: 'created_at') required String createdAt,
    ChatMessage? message,
    required bool done,
    @JsonKey(name: 'done_reason') String? doneReason,

    // Final response fields (when done=True)
    @JsonKey(name: 'total_duration') int? totalDuration,
    @JsonKey(name: 'load_duration') int? loadDuration,
    @JsonKey(name: 'prompt_eval_count') int? promptEvalCount,
    @JsonKey(name: 'prompt_eval_duration') int? promptEvalDuration,
    @JsonKey(name: 'eval_count') int? evalCount,
    @JsonKey(name: 'eval_duration') int? evalDuration,

    // Custom internal fields
    @Default(true) bool success,
    String? error,
  }) = _ChatResponse;

  factory ChatResponse.fromJson(Map<String, dynamic> json) => _$ChatResponseFromJson(json);
}
