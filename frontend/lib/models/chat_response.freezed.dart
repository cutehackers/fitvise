// GENERATED CODE - DO NOT MODIFY BY HAND
// coverage:ignore-file
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'chat_response.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

// dart format off
T _$identity<T>(T value) => value;

/// @nodoc
mixin _$ChatResponse {

 String get model;@JsonKey(name: 'created_at') String get createdAt; ChatMessage? get message; bool get done;@JsonKey(name: 'done_reason') String? get doneReason;// Final response fields (when done=True)
@JsonKey(name: 'total_duration') int? get totalDuration;@JsonKey(name: 'load_duration') int? get loadDuration;@JsonKey(name: 'prompt_eval_count') int? get promptEvalCount;@JsonKey(name: 'prompt_eval_duration') int? get promptEvalDuration;@JsonKey(name: 'eval_count') int? get evalCount;@JsonKey(name: 'eval_duration') int? get evalDuration;// Custom internal fields
 bool get success; String? get error;
/// Create a copy of ChatResponse
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
$ChatResponseCopyWith<ChatResponse> get copyWith => _$ChatResponseCopyWithImpl<ChatResponse>(this as ChatResponse, _$identity);

  /// Serializes this ChatResponse to a JSON map.
  Map<String, dynamic> toJson();


@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is ChatResponse&&(identical(other.model, model) || other.model == model)&&(identical(other.createdAt, createdAt) || other.createdAt == createdAt)&&(identical(other.message, message) || other.message == message)&&(identical(other.done, done) || other.done == done)&&(identical(other.doneReason, doneReason) || other.doneReason == doneReason)&&(identical(other.totalDuration, totalDuration) || other.totalDuration == totalDuration)&&(identical(other.loadDuration, loadDuration) || other.loadDuration == loadDuration)&&(identical(other.promptEvalCount, promptEvalCount) || other.promptEvalCount == promptEvalCount)&&(identical(other.promptEvalDuration, promptEvalDuration) || other.promptEvalDuration == promptEvalDuration)&&(identical(other.evalCount, evalCount) || other.evalCount == evalCount)&&(identical(other.evalDuration, evalDuration) || other.evalDuration == evalDuration)&&(identical(other.success, success) || other.success == success)&&(identical(other.error, error) || other.error == error));
}

@JsonKey(includeFromJson: false, includeToJson: false)
@override
int get hashCode => Object.hash(runtimeType,model,createdAt,message,done,doneReason,totalDuration,loadDuration,promptEvalCount,promptEvalDuration,evalCount,evalDuration,success,error);

@override
String toString() {
  return 'ChatResponse(model: $model, createdAt: $createdAt, message: $message, done: $done, doneReason: $doneReason, totalDuration: $totalDuration, loadDuration: $loadDuration, promptEvalCount: $promptEvalCount, promptEvalDuration: $promptEvalDuration, evalCount: $evalCount, evalDuration: $evalDuration, success: $success, error: $error)';
}


}

/// @nodoc
abstract mixin class $ChatResponseCopyWith<$Res>  {
  factory $ChatResponseCopyWith(ChatResponse value, $Res Function(ChatResponse) _then) = _$ChatResponseCopyWithImpl;
@useResult
$Res call({
 String model,@JsonKey(name: 'created_at') String createdAt, ChatMessage? message, bool done,@JsonKey(name: 'done_reason') String? doneReason,@JsonKey(name: 'total_duration') int? totalDuration,@JsonKey(name: 'load_duration') int? loadDuration,@JsonKey(name: 'prompt_eval_count') int? promptEvalCount,@JsonKey(name: 'prompt_eval_duration') int? promptEvalDuration,@JsonKey(name: 'eval_count') int? evalCount,@JsonKey(name: 'eval_duration') int? evalDuration, bool success, String? error
});


$ChatMessageCopyWith<$Res>? get message;

}
/// @nodoc
class _$ChatResponseCopyWithImpl<$Res>
    implements $ChatResponseCopyWith<$Res> {
  _$ChatResponseCopyWithImpl(this._self, this._then);

  final ChatResponse _self;
  final $Res Function(ChatResponse) _then;

/// Create a copy of ChatResponse
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') @override $Res call({Object? model = null,Object? createdAt = null,Object? message = freezed,Object? done = null,Object? doneReason = freezed,Object? totalDuration = freezed,Object? loadDuration = freezed,Object? promptEvalCount = freezed,Object? promptEvalDuration = freezed,Object? evalCount = freezed,Object? evalDuration = freezed,Object? success = null,Object? error = freezed,}) {
  return _then(_self.copyWith(
model: null == model ? _self.model : model // ignore: cast_nullable_to_non_nullable
as String,createdAt: null == createdAt ? _self.createdAt : createdAt // ignore: cast_nullable_to_non_nullable
as String,message: freezed == message ? _self.message : message // ignore: cast_nullable_to_non_nullable
as ChatMessage?,done: null == done ? _self.done : done // ignore: cast_nullable_to_non_nullable
as bool,doneReason: freezed == doneReason ? _self.doneReason : doneReason // ignore: cast_nullable_to_non_nullable
as String?,totalDuration: freezed == totalDuration ? _self.totalDuration : totalDuration // ignore: cast_nullable_to_non_nullable
as int?,loadDuration: freezed == loadDuration ? _self.loadDuration : loadDuration // ignore: cast_nullable_to_non_nullable
as int?,promptEvalCount: freezed == promptEvalCount ? _self.promptEvalCount : promptEvalCount // ignore: cast_nullable_to_non_nullable
as int?,promptEvalDuration: freezed == promptEvalDuration ? _self.promptEvalDuration : promptEvalDuration // ignore: cast_nullable_to_non_nullable
as int?,evalCount: freezed == evalCount ? _self.evalCount : evalCount // ignore: cast_nullable_to_non_nullable
as int?,evalDuration: freezed == evalDuration ? _self.evalDuration : evalDuration // ignore: cast_nullable_to_non_nullable
as int?,success: null == success ? _self.success : success // ignore: cast_nullable_to_non_nullable
as bool,error: freezed == error ? _self.error : error // ignore: cast_nullable_to_non_nullable
as String?,
  ));
}
/// Create a copy of ChatResponse
/// with the given fields replaced by the non-null parameter values.
@override
@pragma('vm:prefer-inline')
$ChatMessageCopyWith<$Res>? get message {
    if (_self.message == null) {
    return null;
  }

  return $ChatMessageCopyWith<$Res>(_self.message!, (value) {
    return _then(_self.copyWith(message: value));
  });
}
}


/// Adds pattern-matching-related methods to [ChatResponse].
extension ChatResponsePatterns on ChatResponse {
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

@optionalTypeArgs TResult maybeMap<TResult extends Object?>(TResult Function( _ChatResponse value)?  $default,{required TResult orElse(),}){
final _that = this;
switch (_that) {
case _ChatResponse() when $default != null:
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

@optionalTypeArgs TResult map<TResult extends Object?>(TResult Function( _ChatResponse value)  $default,){
final _that = this;
switch (_that) {
case _ChatResponse():
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

@optionalTypeArgs TResult? mapOrNull<TResult extends Object?>(TResult? Function( _ChatResponse value)?  $default,){
final _that = this;
switch (_that) {
case _ChatResponse() when $default != null:
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

@optionalTypeArgs TResult maybeWhen<TResult extends Object?>(TResult Function( String model, @JsonKey(name: 'created_at')  String createdAt,  ChatMessage? message,  bool done, @JsonKey(name: 'done_reason')  String? doneReason, @JsonKey(name: 'total_duration')  int? totalDuration, @JsonKey(name: 'load_duration')  int? loadDuration, @JsonKey(name: 'prompt_eval_count')  int? promptEvalCount, @JsonKey(name: 'prompt_eval_duration')  int? promptEvalDuration, @JsonKey(name: 'eval_count')  int? evalCount, @JsonKey(name: 'eval_duration')  int? evalDuration,  bool success,  String? error)?  $default,{required TResult orElse(),}) {final _that = this;
switch (_that) {
case _ChatResponse() when $default != null:
return $default(_that.model,_that.createdAt,_that.message,_that.done,_that.doneReason,_that.totalDuration,_that.loadDuration,_that.promptEvalCount,_that.promptEvalDuration,_that.evalCount,_that.evalDuration,_that.success,_that.error);case _:
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

@optionalTypeArgs TResult when<TResult extends Object?>(TResult Function( String model, @JsonKey(name: 'created_at')  String createdAt,  ChatMessage? message,  bool done, @JsonKey(name: 'done_reason')  String? doneReason, @JsonKey(name: 'total_duration')  int? totalDuration, @JsonKey(name: 'load_duration')  int? loadDuration, @JsonKey(name: 'prompt_eval_count')  int? promptEvalCount, @JsonKey(name: 'prompt_eval_duration')  int? promptEvalDuration, @JsonKey(name: 'eval_count')  int? evalCount, @JsonKey(name: 'eval_duration')  int? evalDuration,  bool success,  String? error)  $default,) {final _that = this;
switch (_that) {
case _ChatResponse():
return $default(_that.model,_that.createdAt,_that.message,_that.done,_that.doneReason,_that.totalDuration,_that.loadDuration,_that.promptEvalCount,_that.promptEvalDuration,_that.evalCount,_that.evalDuration,_that.success,_that.error);case _:
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

@optionalTypeArgs TResult? whenOrNull<TResult extends Object?>(TResult? Function( String model, @JsonKey(name: 'created_at')  String createdAt,  ChatMessage? message,  bool done, @JsonKey(name: 'done_reason')  String? doneReason, @JsonKey(name: 'total_duration')  int? totalDuration, @JsonKey(name: 'load_duration')  int? loadDuration, @JsonKey(name: 'prompt_eval_count')  int? promptEvalCount, @JsonKey(name: 'prompt_eval_duration')  int? promptEvalDuration, @JsonKey(name: 'eval_count')  int? evalCount, @JsonKey(name: 'eval_duration')  int? evalDuration,  bool success,  String? error)?  $default,) {final _that = this;
switch (_that) {
case _ChatResponse() when $default != null:
return $default(_that.model,_that.createdAt,_that.message,_that.done,_that.doneReason,_that.totalDuration,_that.loadDuration,_that.promptEvalCount,_that.promptEvalDuration,_that.evalCount,_that.evalDuration,_that.success,_that.error);case _:
  return null;

}
}

}

/// @nodoc
@JsonSerializable()

class _ChatResponse implements ChatResponse {
  const _ChatResponse({required this.model, @JsonKey(name: 'created_at') required this.createdAt, this.message, required this.done, @JsonKey(name: 'done_reason') this.doneReason, @JsonKey(name: 'total_duration') this.totalDuration, @JsonKey(name: 'load_duration') this.loadDuration, @JsonKey(name: 'prompt_eval_count') this.promptEvalCount, @JsonKey(name: 'prompt_eval_duration') this.promptEvalDuration, @JsonKey(name: 'eval_count') this.evalCount, @JsonKey(name: 'eval_duration') this.evalDuration, this.success = true, this.error});
  factory _ChatResponse.fromJson(Map<String, dynamic> json) => _$ChatResponseFromJson(json);

@override final  String model;
@override@JsonKey(name: 'created_at') final  String createdAt;
@override final  ChatMessage? message;
@override final  bool done;
@override@JsonKey(name: 'done_reason') final  String? doneReason;
// Final response fields (when done=True)
@override@JsonKey(name: 'total_duration') final  int? totalDuration;
@override@JsonKey(name: 'load_duration') final  int? loadDuration;
@override@JsonKey(name: 'prompt_eval_count') final  int? promptEvalCount;
@override@JsonKey(name: 'prompt_eval_duration') final  int? promptEvalDuration;
@override@JsonKey(name: 'eval_count') final  int? evalCount;
@override@JsonKey(name: 'eval_duration') final  int? evalDuration;
// Custom internal fields
@override@JsonKey() final  bool success;
@override final  String? error;

/// Create a copy of ChatResponse
/// with the given fields replaced by the non-null parameter values.
@override @JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
_$ChatResponseCopyWith<_ChatResponse> get copyWith => __$ChatResponseCopyWithImpl<_ChatResponse>(this, _$identity);

@override
Map<String, dynamic> toJson() {
  return _$ChatResponseToJson(this, );
}

@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _ChatResponse&&(identical(other.model, model) || other.model == model)&&(identical(other.createdAt, createdAt) || other.createdAt == createdAt)&&(identical(other.message, message) || other.message == message)&&(identical(other.done, done) || other.done == done)&&(identical(other.doneReason, doneReason) || other.doneReason == doneReason)&&(identical(other.totalDuration, totalDuration) || other.totalDuration == totalDuration)&&(identical(other.loadDuration, loadDuration) || other.loadDuration == loadDuration)&&(identical(other.promptEvalCount, promptEvalCount) || other.promptEvalCount == promptEvalCount)&&(identical(other.promptEvalDuration, promptEvalDuration) || other.promptEvalDuration == promptEvalDuration)&&(identical(other.evalCount, evalCount) || other.evalCount == evalCount)&&(identical(other.evalDuration, evalDuration) || other.evalDuration == evalDuration)&&(identical(other.success, success) || other.success == success)&&(identical(other.error, error) || other.error == error));
}

@JsonKey(includeFromJson: false, includeToJson: false)
@override
int get hashCode => Object.hash(runtimeType,model,createdAt,message,done,doneReason,totalDuration,loadDuration,promptEvalCount,promptEvalDuration,evalCount,evalDuration,success,error);

@override
String toString() {
  return 'ChatResponse(model: $model, createdAt: $createdAt, message: $message, done: $done, doneReason: $doneReason, totalDuration: $totalDuration, loadDuration: $loadDuration, promptEvalCount: $promptEvalCount, promptEvalDuration: $promptEvalDuration, evalCount: $evalCount, evalDuration: $evalDuration, success: $success, error: $error)';
}


}

/// @nodoc
abstract mixin class _$ChatResponseCopyWith<$Res> implements $ChatResponseCopyWith<$Res> {
  factory _$ChatResponseCopyWith(_ChatResponse value, $Res Function(_ChatResponse) _then) = __$ChatResponseCopyWithImpl;
@override @useResult
$Res call({
 String model,@JsonKey(name: 'created_at') String createdAt, ChatMessage? message, bool done,@JsonKey(name: 'done_reason') String? doneReason,@JsonKey(name: 'total_duration') int? totalDuration,@JsonKey(name: 'load_duration') int? loadDuration,@JsonKey(name: 'prompt_eval_count') int? promptEvalCount,@JsonKey(name: 'prompt_eval_duration') int? promptEvalDuration,@JsonKey(name: 'eval_count') int? evalCount,@JsonKey(name: 'eval_duration') int? evalDuration, bool success, String? error
});


@override $ChatMessageCopyWith<$Res>? get message;

}
/// @nodoc
class __$ChatResponseCopyWithImpl<$Res>
    implements _$ChatResponseCopyWith<$Res> {
  __$ChatResponseCopyWithImpl(this._self, this._then);

  final _ChatResponse _self;
  final $Res Function(_ChatResponse) _then;

/// Create a copy of ChatResponse
/// with the given fields replaced by the non-null parameter values.
@override @pragma('vm:prefer-inline') $Res call({Object? model = null,Object? createdAt = null,Object? message = freezed,Object? done = null,Object? doneReason = freezed,Object? totalDuration = freezed,Object? loadDuration = freezed,Object? promptEvalCount = freezed,Object? promptEvalDuration = freezed,Object? evalCount = freezed,Object? evalDuration = freezed,Object? success = null,Object? error = freezed,}) {
  return _then(_ChatResponse(
model: null == model ? _self.model : model // ignore: cast_nullable_to_non_nullable
as String,createdAt: null == createdAt ? _self.createdAt : createdAt // ignore: cast_nullable_to_non_nullable
as String,message: freezed == message ? _self.message : message // ignore: cast_nullable_to_non_nullable
as ChatMessage?,done: null == done ? _self.done : done // ignore: cast_nullable_to_non_nullable
as bool,doneReason: freezed == doneReason ? _self.doneReason : doneReason // ignore: cast_nullable_to_non_nullable
as String?,totalDuration: freezed == totalDuration ? _self.totalDuration : totalDuration // ignore: cast_nullable_to_non_nullable
as int?,loadDuration: freezed == loadDuration ? _self.loadDuration : loadDuration // ignore: cast_nullable_to_non_nullable
as int?,promptEvalCount: freezed == promptEvalCount ? _self.promptEvalCount : promptEvalCount // ignore: cast_nullable_to_non_nullable
as int?,promptEvalDuration: freezed == promptEvalDuration ? _self.promptEvalDuration : promptEvalDuration // ignore: cast_nullable_to_non_nullable
as int?,evalCount: freezed == evalCount ? _self.evalCount : evalCount // ignore: cast_nullable_to_non_nullable
as int?,evalDuration: freezed == evalDuration ? _self.evalDuration : evalDuration // ignore: cast_nullable_to_non_nullable
as int?,success: null == success ? _self.success : success // ignore: cast_nullable_to_non_nullable
as bool,error: freezed == error ? _self.error : error // ignore: cast_nullable_to_non_nullable
as String?,
  ));
}

/// Create a copy of ChatResponse
/// with the given fields replaced by the non-null parameter values.
@override
@pragma('vm:prefer-inline')
$ChatMessageCopyWith<$Res>? get message {
    if (_self.message == null) {
    return null;
  }

  return $ChatMessageCopyWith<$Res>(_self.message!, (value) {
    return _then(_self.copyWith(message: value));
  });
}
}

// dart format on
