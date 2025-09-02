enum MessageRole { user, ai, system }

class Message {
  final String id;
  final MessageRole role; // 'user' or 'ai', 'system'
  final String text;
  final DateTime timestamp;
  final String type; // 'text', 'image', 'file'
  final List<MessageAction>? actions;
  final bool isEdited;
  final bool isStreaming; // New property to track streaming state

  const Message({
    required this.role,
    required this.text,
    required this.timestamp,
    this.id = '',
    this.type = 'text',
    this.actions,
    this.isEdited = false,
    this.isStreaming = false, // Default to false for completed messages
  });

  const Message.user({
    required this.text,
    required this.timestamp,
    this.id = '',
    this.type = 'text',
    this.actions,
    this.isEdited = false,
    this.isStreaming = false, // Default to false for completed messages
  }) : role = MessageRole.user;

  const Message.ai({
    required this.text,
    required this.timestamp,
    this.id = '',
    this.type = 'text',
    this.actions,
    this.isEdited = false,
    this.isStreaming = false, // Default to false for completed messages
  }) : role = MessageRole.ai;

  Message copyWith({
    String? id,
    MessageRole? role,
    String? text,
    DateTime? timestamp,
    String? type,
    List<MessageAction>? actions,
    bool? isEdited,
    bool? isStreaming,
  }) {
    return Message(
      id: id ?? this.id,
      role: role ?? this.role,
      text: text ?? this.text,
      timestamp: timestamp ?? this.timestamp,
      type: type ?? this.type,
      actions: actions ?? this.actions,
      isEdited: isEdited ?? this.isEdited,
      isStreaming: isStreaming ?? this.isStreaming,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'role': role,
      'text': text,
      'timestamp': timestamp.toIso8601String(),
      'type': type,
      'actions': actions?.map((action) => action.toJson()).toList(),
      'isEdited': isEdited,
      'isStreaming': isStreaming,
    };
  }

  factory Message.fromJson(Map<String, dynamic> json) {
    return Message(
      id: json['id'],
      role: json['role'],
      text: json['text'],
      timestamp: DateTime.parse(json['timestamp']),
      type: json['type'] ?? 'text',
      actions: json['actions'] != null
          ? (json['actions'] as List)
                .map((action) => MessageAction.fromJson(action))
                .toList()
          : null,
      isEdited: json['isEdited'] ?? false,
      isStreaming: json['isStreaming'] ?? false,
    );
  }
}

class MessageAction {
  final String label;
  final String action;

  MessageAction({required this.label, required this.action});

  Map<String, dynamic> toJson() {
    return {'label': label, 'action': action};
  }

  factory MessageAction.fromJson(Map<String, dynamic> json) {
    return MessageAction(label: json['label'], action: json['action']);
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
