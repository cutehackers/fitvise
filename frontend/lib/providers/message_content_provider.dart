import 'package:flutter_riverpod/flutter_riverpod.dart';

final messageContentProvider = StateNotifierProvider.family<MessageContentNotifier, MessageContentState, String>((
  ref,
  messageId,
) {
  return MessageContentNotifier(messageId);
});

class MessageContentState {
  final String content;
  final bool isStreaming;

  const MessageContentState.empty() : content = '', isStreaming = false;

  const MessageContentState({required this.content, this.isStreaming = false});

  MessageContentState copyWith({String? content, bool? isStreaming}) {
    return MessageContentState(content: content ?? this.content, isStreaming: isStreaming ?? this.isStreaming);
  }
}

class MessageContentNotifier extends StateNotifier<MessageContentState> {
  MessageContentNotifier(this.messageId) : super(MessageContentState.empty());

  final String messageId;

  /// message id: message stream controller
  // final LinkedHashMap<String, StreamController<String>> _messageStreamMap = LinkedHashMap();

  void appendToMessageContent(String chunk) {
    state = state.copyWith(content: state.content + chunk, isStreaming: true);
  }

  void completeMessageContent() {
    state = state.copyWith(isStreaming: false);
  }

  // @override
  // void dispose() {
  //   // Close all stream controllers
  //   for (final controller in _messageStreamMap.values) {
  //     if (!controller.isClosed) {
  //       controller.close();
  //     }
  //   }
  //   _messageStreamMap.clear();
  //   super.dispose();
  // }
}
