class Message {
  final String id;
  final String sender; // 'user' or 'ai'
  final String text;
  final DateTime timestamp;
  final String type; // 'text', 'image', 'file'
  final List<MessageAction>? actions;
  final bool isEdited;

  Message({
    required this.id,
    required this.sender,
    required this.text,
    required this.timestamp,
    this.type = 'text',
    this.actions,
    this.isEdited = false,
  });

  Message copyWith({
    String? id,
    String? sender,
    String? text,
    DateTime? timestamp,
    String? type,
    List<MessageAction>? actions,
    bool? isEdited,
  }) {
    return Message(
      id: id ?? this.id,
      sender: sender ?? this.sender,
      text: text ?? this.text,
      timestamp: timestamp ?? this.timestamp,
      type: type ?? this.type,
      actions: actions ?? this.actions,
      isEdited: isEdited ?? this.isEdited,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'sender': sender,
      'text': text,
      'timestamp': timestamp.toIso8601String(),
      'type': type,
      'actions': actions?.map((action) => action.toJson()).toList(),
      'isEdited': isEdited,
    };
  }

  factory Message.fromJson(Map<String, dynamic> json) {
    return Message(
      id: json['id'],
      sender: json['sender'],
      text: json['text'],
      timestamp: DateTime.parse(json['timestamp']),
      type: json['type'] ?? 'text',
      actions: json['actions'] != null
          ? (json['actions'] as List)
              .map((action) => MessageAction.fromJson(action))
              .toList()
          : null,
      isEdited: json['isEdited'] ?? false,
    );
  }
}

class MessageAction {
  final String label;
  final String action;

  MessageAction({
    required this.label,
    required this.action,
  });

  Map<String, dynamic> toJson() {
    return {
      'label': label,
      'action': action,
    };
  }

  factory MessageAction.fromJson(Map<String, dynamic> json) {
    return MessageAction(
      label: json['label'],
      action: json['action'],
    );
  }
}

class WelcomePrompt {
  final String icon;
  final String text;
  final String category;

  WelcomePrompt({
    required this.icon,
    required this.text,
    required this.category,
  });
}