import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/chat_provider.dart';

class QuickReplies extends StatelessWidget {
  const QuickReplies({super.key, required this.sessionId});

  final String sessionId;

  @override
  Widget build(BuildContext context) {
    final chatProvider = Provider.of<ChatProvider>(context, listen: false);

    return Container(
      margin: const EdgeInsets.symmetric(vertical: 16),
      child: Wrap(
        alignment: WrapAlignment.center,
        spacing: 8,
        runSpacing: 8,
        children: chatProvider.quickReplies.map((reply) {
          return _QuickReplyChip(text: reply, onTap: () => chatProvider.sendQuickReply(sessionId, reply));
        }).toList(),
      ),
    );
  }
}

class _QuickReplyChip extends StatelessWidget {
  final String text;
  final VoidCallback onTap;

  const _QuickReplyChip({required this.text, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return ActionChip(
      label: Text(text),
      onPressed: onTap,
      backgroundColor: Colors.transparent,
      side: BorderSide(color: Theme.of(context).dividerColor),
      labelStyle: Theme.of(context).textTheme.bodyMedium,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
    );
  }
}
