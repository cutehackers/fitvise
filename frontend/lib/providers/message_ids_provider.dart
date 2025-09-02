import 'package:flutter_riverpod/flutter_riverpod.dart';

final messageIdsProvider =
    StateNotifierProvider<MessageIdsNotifier, List<String>>((ref) {
      return MessageIdsNotifier();
    });

class MessageIdsNotifier extends StateNotifier<List<String>> {
  MessageIdsNotifier() : super([]);

  void add(String msgId) {
    state = [...state, msgId];
  }

  void remove(String msgId) {
    state = state.where((id) => id != msgId).toList();
  }

  void clear() {
    state = [];
  }
}
