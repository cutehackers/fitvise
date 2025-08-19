// GENERATED CODE - DO NOT MODIFY BY HAND
// coverage:ignore-file
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'chat_request.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

// dart format off
T _$identity<T>(T value) => value;

/// @nodoc
mixin _$ChatRequest {

 String get model;@JsonKey(name: 'session_id') String get sessionId; ChatMessage get message; List<Map<String, dynamic>>? get tools; bool get think; String? get format; Map<String, dynamic>? get options; bool get stream;@JsonKey(name: 'keep_alive') String? get keepAlive;
/// Create a copy of ChatRequest
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
$ChatRequestCopyWith<ChatRequest> get copyWith => _$ChatRequestCopyWithImpl<ChatRequest>(this as ChatRequest, _$identity);

  /// Serializes this ChatRequest to a JSON map.
  Map<String, dynamic> toJson();


@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is ChatRequest&&(identical(other.model, model) || other.model == model)&&(identical(other.sessionId, sessionId) || other.sessionId == sessionId)&&(identical(other.message, message) || other.message == message)&&const DeepCollectionEquality().equals(other.tools, tools)&&(identical(other.think, think) || other.think == think)&&(identical(other.format, format) || other.format == format)&&const DeepCollectionEquality().equals(other.options, options)&&(identical(other.stream, stream) || other.stream == stream)&&(identical(other.keepAlive, keepAlive) || other.keepAlive == keepAlive));
}

@JsonKey(includeFromJson: false, includeToJson: false)
@override
int get hashCode => Object.hash(runtimeType,model,sessionId,message,const DeepCollectionEquality().hash(tools),think,format,const DeepCollectionEquality().hash(options),stream,keepAlive);

@override
String toString() {
  return 'ChatRequest(model: $model, sessionId: $sessionId, message: $message, tools: $tools, think: $think, format: $format, options: $options, stream: $stream, keepAlive: $keepAlive)';
}


}

/// @nodoc
abstract mixin class $ChatRequestCopyWith<$Res>  {
  factory $ChatRequestCopyWith(ChatRequest value, $Res Function(ChatRequest) _then) = _$ChatRequestCopyWithImpl;
@useResult
$Res call({
 String model,@JsonKey(name: 'session_id') String sessionId, ChatMessage message, List<Map<String, dynamic>>? tools, bool think, String? format, Map<String, dynamic>? options, bool stream,@JsonKey(name: 'keep_alive') String? keepAlive
});


$ChatMessageCopyWith<$Res> get message;

}
/// @nodoc
class _$ChatRequestCopyWithImpl<$Res>
    implements $ChatRequestCopyWith<$Res> {
  _$ChatRequestCopyWithImpl(this._self, this._then);

  final ChatRequest _self;
  final $Res Function(ChatRequest) _then;

/// Create a copy of ChatRequest
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') @override $Res call({Object? model = null,Object? sessionId = null,Object? message = null,Object? tools = freezed,Object? think = null,Object? format = freezed,Object? options = freezed,Object? stream = null,Object? keepAlive = freezed,}) {
  return _then(_self.copyWith(
model: null == model ? _self.model : model // ignore: cast_nullable_to_non_nullable
as String,sessionId: null == sessionId ? _self.sessionId : sessionId // ignore: cast_nullable_to_non_nullable
as String,message: null == message ? _self.message : message // ignore: cast_nullable_to_non_nullable
as ChatMessage,tools: freezed == tools ? _self.tools : tools // ignore: cast_nullable_to_non_nullable
as List<Map<String, dynamic>>?,think: null == think ? _self.think : think // ignore: cast_nullable_to_non_nullable
as bool,format: freezed == format ? _self.format : format // ignore: cast_nullable_to_non_nullable
as String?,options: freezed == options ? _self.options : options // ignore: cast_nullable_to_non_nullable
as Map<String, dynamic>?,stream: null == stream ? _self.stream : stream // ignore: cast_nullable_to_non_nullable
as bool,keepAlive: freezed == keepAlive ? _self.keepAlive : keepAlive // ignore: cast_nullable_to_non_nullable
as String?,
  ));
}
/// Create a copy of ChatRequest
/// with the given fields replaced by the non-null parameter values.
@override
@pragma('vm:prefer-inline')
$ChatMessageCopyWith<$Res> get message {
  
  return $ChatMessageCopyWith<$Res>(_self.message, (value) {
    return _then(_self.copyWith(message: value));
  });
}
}


/// Adds pattern-matching-related methods to [ChatRequest].
extension ChatRequestPatterns on ChatRequest {
/// A variant of `map` that fallback to returning `orElse`.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case _:
///     return orElse();
/// }
/// ```

@optionalTypeArgs TResult maybeMap<TResult extends Object?>(TResult Function( _ChatRequest value)?  $default,{required TResult orElse(),}){
final _that = this;
switch (_that) {
case _ChatRequest() when $default != null:
return $default(_that);case _:
  return orElse();

}
}
/// A `switch`-like method, using callbacks.
///
/// Callbacks receives the raw object, upcasted.
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case final Subclass2 value:
///     return ...;
/// }
/// ```

@optionalTypeArgs TResult map<TResult extends Object?>(TResult Function( _ChatRequest value)  $default,){
final _that = this;
switch (_that) {
case _ChatRequest():
return $default(_that);case _:
  throw StateError('Unexpected subclass');

}
}
/// A variant of `map` that fallback to returning `null`.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case _:
///     return null;
/// }
/// ```

@optionalTypeArgs TResult? mapOrNull<TResult extends Object?>(TResult? Function( _ChatRequest value)?  $default,){
final _that = this;
switch (_that) {
case _ChatRequest() when $default != null:
return $default(_that);case _:
  return null;

}
}
/// A variant of `when` that fallback to an `orElse` callback.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case _:
///     return orElse();
/// }
/// ```

@optionalTypeArgs TResult maybeWhen<TResult extends Object?>(TResult Function( String model, @JsonKey(name: 'session_id')  String sessionId,  ChatMessage message,  List<Map<String, dynamic>>? tools,  bool think,  String? format,  Map<String, dynamic>? options,  bool stream, @JsonKey(name: 'keep_alive')  String? keepAlive)?  $default,{required TResult orElse(),}) {final _that = this;
switch (_that) {
case _ChatRequest() when $default != null:
return $default(_that.model,_that.sessionId,_that.message,_that.tools,_that.think,_that.format,_that.options,_that.stream,_that.keepAlive);case _:
  return orElse();

}
}
/// A `switch`-like method, using callbacks.
///
/// As opposed to `map`, this offers destructuring.
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case Subclass2(:final field2):
///     return ...;
/// }
/// ```

@optionalTypeArgs TResult when<TResult extends Object?>(TResult Function( String model, @JsonKey(name: 'session_id')  String sessionId,  ChatMessage message,  List<Map<String, dynamic>>? tools,  bool think,  String? format,  Map<String, dynamic>? options,  bool stream, @JsonKey(name: 'keep_alive')  String? keepAlive)  $default,) {final _that = this;
switch (_that) {
case _ChatRequest():
return $default(_that.model,_that.sessionId,_that.message,_that.tools,_that.think,_that.format,_that.options,_that.stream,_that.keepAlive);case _:
  throw StateError('Unexpected subclass');

}
}
/// A variant of `when` that fallback to returning `null`
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case _:
///     return null;
/// }
/// ```

@optionalTypeArgs TResult? whenOrNull<TResult extends Object?>(TResult? Function( String model, @JsonKey(name: 'session_id')  String sessionId,  ChatMessage message,  List<Map<String, dynamic>>? tools,  bool think,  String? format,  Map<String, dynamic>? options,  bool stream, @JsonKey(name: 'keep_alive')  String? keepAlive)?  $default,) {final _that = this;
switch (_that) {
case _ChatRequest() when $default != null:
return $default(_that.model,_that.sessionId,_that.message,_that.tools,_that.think,_that.format,_that.options,_that.stream,_that.keepAlive);case _:
  return null;

}
}

}

/// @nodoc
@JsonSerializable()

class _ChatRequest implements ChatRequest {
  const _ChatRequest({required this.model, @JsonKey(name: 'session_id') required this.sessionId, required this.message, final  List<Map<String, dynamic>>? tools, this.think = false, this.format, final  Map<String, dynamic>? options, this.stream = true, @JsonKey(name: 'keep_alive') this.keepAlive}): _tools = tools,_options = options;
  factory _ChatRequest.fromJson(Map<String, dynamic> json) => _$ChatRequestFromJson(json);

@override final  String model;
@override@JsonKey(name: 'session_id') final  String sessionId;
@override final  ChatMessage message;
 final  List<Map<String, dynamic>>? _tools;
@override List<Map<String, dynamic>>? get tools {
  final value = _tools;
  if (value == null) return null;
  if (_tools is EqualUnmodifiableListView) return _tools;
  // ignore: implicit_dynamic_type
  return EqualUnmodifiableListView(value);
}

@override@JsonKey() final  bool think;
@override final  String? format;
 final  Map<String, dynamic>? _options;
@override Map<String, dynamic>? get options {
  final value = _options;
  if (value == null) return null;
  if (_options is EqualUnmodifiableMapView) return _options;
  // ignore: implicit_dynamic_type
  return EqualUnmodifiableMapView(value);
}

@override@JsonKey() final  bool stream;
@override@JsonKey(name: 'keep_alive') final  String? keepAlive;

/// Create a copy of ChatRequest
/// with the given fields replaced by the non-null parameter values.
@override @JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
_$ChatRequestCopyWith<_ChatRequest> get copyWith => __$ChatRequestCopyWithImpl<_ChatRequest>(this, _$identity);

@override
Map<String, dynamic> toJson() {
  return _$ChatRequestToJson(this, );
}

@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _ChatRequest&&(identical(other.model, model) || other.model == model)&&(identical(other.sessionId, sessionId) || other.sessionId == sessionId)&&(identical(other.message, message) || other.message == message)&&const DeepCollectionEquality().equals(other._tools, _tools)&&(identical(other.think, think) || other.think == think)&&(identical(other.format, format) || other.format == format)&&const DeepCollectionEquality().equals(other._options, _options)&&(identical(other.stream, stream) || other.stream == stream)&&(identical(other.keepAlive, keepAlive) || other.keepAlive == keepAlive));
}

@JsonKey(includeFromJson: false, includeToJson: false)
@override
int get hashCode => Object.hash(runtimeType,model,sessionId,message,const DeepCollectionEquality().hash(_tools),think,format,const DeepCollectionEquality().hash(_options),stream,keepAlive);

@override
String toString() {
  return 'ChatRequest(model: $model, sessionId: $sessionId, message: $message, tools: $tools, think: $think, format: $format, options: $options, stream: $stream, keepAlive: $keepAlive)';
}


}

/// @nodoc
abstract mixin class _$ChatRequestCopyWith<$Res> implements $ChatRequestCopyWith<$Res> {
  factory _$ChatRequestCopyWith(_ChatRequest value, $Res Function(_ChatRequest) _then) = __$ChatRequestCopyWithImpl;
@override @useResult
$Res call({
 String model,@JsonKey(name: 'session_id') String sessionId, ChatMessage message, List<Map<String, dynamic>>? tools, bool think, String? format, Map<String, dynamic>? options, bool stream,@JsonKey(name: 'keep_alive') String? keepAlive
});


@override $ChatMessageCopyWith<$Res> get message;

}
/// @nodoc
class __$ChatRequestCopyWithImpl<$Res>
    implements _$ChatRequestCopyWith<$Res> {
  __$ChatRequestCopyWithImpl(this._self, this._then);

  final _ChatRequest _self;
  final $Res Function(_ChatRequest) _then;

/// Create a copy of ChatRequest
/// with the given fields replaced by the non-null parameter values.
@override @pragma('vm:prefer-inline') $Res call({Object? model = null,Object? sessionId = null,Object? message = null,Object? tools = freezed,Object? think = null,Object? format = freezed,Object? options = freezed,Object? stream = null,Object? keepAlive = freezed,}) {
  return _then(_ChatRequest(
model: null == model ? _self.model : model // ignore: cast_nullable_to_non_nullable
as String,sessionId: null == sessionId ? _self.sessionId : sessionId // ignore: cast_nullable_to_non_nullable
as String,message: null == message ? _self.message : message // ignore: cast_nullable_to_non_nullable
as ChatMessage,tools: freezed == tools ? _self._tools : tools // ignore: cast_nullable_to_non_nullable
as List<Map<String, dynamic>>?,think: null == think ? _self.think : think // ignore: cast_nullable_to_non_nullable
as bool,format: freezed == format ? _self.format : format // ignore: cast_nullable_to_non_nullable
as String?,options: freezed == options ? _self._options : options // ignore: cast_nullable_to_non_nullable
as Map<String, dynamic>?,stream: null == stream ? _self.stream : stream // ignore: cast_nullable_to_non_nullable
as bool,keepAlive: freezed == keepAlive ? _self.keepAlive : keepAlive // ignore: cast_nullable_to_non_nullable
as String?,
  ));
}

/// Create a copy of ChatRequest
/// with the given fields replaced by the non-null parameter values.
@override
@pragma('vm:prefer-inline')
$ChatMessageCopyWith<$Res> get message {
  
  return $ChatMessageCopyWith<$Res>(_self.message, (value) {
    return _then(_self.copyWith(message: value));
  });
}
}

// dart format on
