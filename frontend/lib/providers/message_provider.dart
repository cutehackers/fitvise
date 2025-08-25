import 'package:fitvise/models/message.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

final messageProvider = StateNotifierProvider.family<MessageProvider, Message, String>(
  (ref, messageId) =>
      MessageProvider(Message(id: messageId, role: MessageRole.user, text: '', timestamp: DateTime.now())),
);

class MessageProvider extends StateNotifier<Message> {
  MessageProvider(super.message);

  void update({String? text, bool? isStreaming, List<MessageAction>? actions}) {
    state = state.copyWith(
      text: text ?? state.text,
      isStreaming: isStreaming ?? state.isStreaming,
      actions: actions ?? state.actions,
    );
  }

  void updateContent(String content) {
    state = state.copyWith(text: content);
  }

  void appendChunkToContent(String chunk) {
    state = state.copyWith(text: state.text + chunk);
  }

  void setStreaming(bool isStreaming) {
    state = state.copyWith(isStreaming: isStreaming);
  }

  void setMessage(Message message) {
    state = message;
  }
}
