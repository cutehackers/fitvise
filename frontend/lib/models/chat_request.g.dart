// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'chat_request.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

_ChatRequest _$ChatRequestFromJson(Map<String, dynamic> json) => _ChatRequest(
  model: json['model'] as String,
  sessionId: json['session_id'] as String,
  message: ChatMessage.fromJson(json['message'] as Map<String, dynamic>),
  tools: (json['tools'] as List<dynamic>?)
      ?.map((e) => e as Map<String, dynamic>)
      .toList(),
  think: json['think'] as bool? ?? false,
  format: json['format'] as String?,
  options: json['options'] as Map<String, dynamic>?,
  stream: json['stream'] as bool? ?? true,
  keepAlive: json['keep_alive'] as String?,
);

Map<String, dynamic> _$ChatRequestToJson(_ChatRequest instance) =>
    <String, dynamic>{
      'model': instance.model,
      'session_id': instance.sessionId,
      'message': instance.message,
      'tools': instance.tools,
      'think': instance.think,
      'format': instance.format,
      'options': instance.options,
      'stream': instance.stream,
      'keep_alive': instance.keepAlive,
    };
