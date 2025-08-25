// ignore_for_file: invalid_annotation_target

import 'package:freezed_annotation/freezed_annotation.dart';

part 'chat_message.freezed.dart';
part 'chat_message.g.dart';

/// Single message in a chat conversation
@freezed
abstract class ChatMessage with _$ChatMessage {
  const factory ChatMessage({
    required String role, // 'user' or 'ai', or 'system'
    required String content,
    @Default(false) bool thinking,
    @JsonKey(name: 'images') List<String>? images,
    @JsonKey(name: 'tool_calls') List<Map<String, dynamic>>? toolCalls,
    @JsonKey(name: 'tool_name') String? toolName,
  }) = _ChatMessage;

  factory ChatMessage.fromJson(Map<String, dynamic> json) => _$ChatMessageFromJson(json);
}
