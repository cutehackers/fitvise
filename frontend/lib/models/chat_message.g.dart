// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'chat_message.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

_ChatMessage _$ChatMessageFromJson(Map<String, dynamic> json) => _ChatMessage(
  role: json['role'] as String,
  content: json['content'] as String,
  thinking: json['thinking'] as bool? ?? false,
  images: (json['images'] as List<dynamic>?)?.map((e) => e as String).toList(),
  toolCalls: (json['tool_calls'] as List<dynamic>?)
      ?.map((e) => e as Map<String, dynamic>)
      .toList(),
  toolName: json['tool_name'] as String?,
);

Map<String, dynamic> _$ChatMessageToJson(_ChatMessage instance) =>
    <String, dynamic>{
      'role': instance.role,
      'content': instance.content,
      'thinking': instance.thinking,
      'images': instance.images,
      'tool_calls': instance.toolCalls,
      'tool_name': instance.toolName,
    };
