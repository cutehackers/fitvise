import 'package:fitvise/models/message.dart';
import 'package:fitvise/providers/message_provider.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../theme/app_theme.dart';

/// A highly customizable message bubble widget for chat interfaces
///
/// Features:
/// - Support for both user and AI messages
/// - Customizable appearance with themes
/// - Avatar support with gradients
/// - Action buttons and message editing
/// - Timestamp and status indicators
/// - Accessibility support
/// - Animation support
class MessageBubble extends ConsumerWidget {
  final String messageId;

  /// Configuration for bubble appearance
  final MessageBubbleConfig? config;

  /// Timestamp to display
  final DateTime? timestamp;

  /// Whether the message was edited
  final bool isEdited;

  /// Action buttons to display
  final List<Widget>? actions;

  /// Sender name to display
  final String? senderName;

  /// Custom avatar widget
  final Widget? avatar;

  /// Message status (sent, delivered, read)
  final String? status;

  /// Animation duration for bubble appearance
  final Duration? animationDuration;

  /// Whether to show entrance animation
  final bool animated;

  const MessageBubble({
    required this.messageId,
    this.config,
    this.timestamp,
    this.isEdited = false,
    this.actions,
    this.senderName,
    this.avatar,
    this.status,
    this.animationDuration,
    this.animated = false,
    super.key,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final theme = Theme.of(context);

    final isUser = ref.watch(
      messageProvider(messageId).select((s) => s.role == MessageRole.user),
    );
    final actions = ref.watch(
      messageProvider(messageId).select((s) => s.actions),
    );

    final configs = _getConfigs(theme, isUser);

    Widget bubble = Container(
      margin: configs.margin,
      child: Column(
        crossAxisAlignment: isUser
            ? CrossAxisAlignment.end
            : CrossAxisAlignment.start,
        children: [
          // Sender info (for AI messages)
          if (!isUser && configs.showAvatar)
            _Sender(avatar: avatar, senderName: senderName, config: configs),

          // Message content
          MessageContent(
            messageId: messageId,
            isUser: isUser,
            config: configs,
            isEdited: isEdited,
            messageActions: [
              Consumer(
                builder: (context, ref, child) {
                  final text = ref.watch(
                    messageProvider(messageId).select((s) => s.text),
                  );
                  return ContentAction(
                    icon: Icons.copy_rounded,
                    onPressed: () => _copyToClipboard(context, text),
                  );
                },
              ),
              // user
              // if (isUser)
              // TODO: enable edit action
              // ContentAction(
              //   icon: Icons.copy_rounded,
              //   onPressed: () => ref.read(messageProvider(messageId).notifier).editMessage(id, text),
              // ),
            ],
          ),

          // Action buttons
          if (actions != null && actions.isNotEmpty)
            _ActionButtonContainer(
              actions: actions.map((e) => _ActionButton(action: e)).toList(),
              isUser: isUser,
            ),

          // Timestamp
          if (configs.showTimestamp && timestamp != null)
            _Time(timestamp: timestamp!, isUser: isUser),
        ],
      ),
    );

    if (animated && animationDuration != null) {
      return TweenAnimationBuilder<double>(
        duration: animationDuration!,
        tween: Tween(begin: 0.0, end: 1.0),
        curve: Curves.easeOutCubic,
        builder: (context, value, child) {
          return Transform.translate(
            offset: Offset(isUser ? 50 * (1 - value) : -50 * (1 - value), 0),
            child: Opacity(opacity: value, child: child),
          );
        },
        child: bubble,
      );
    }

    return bubble;
  }

  MessageBubbleConfig _getConfigs(ThemeData theme, bool isUser) {
    final defaultConfig = isUser
        ? _getDefaultUserConfig(theme)
        : _getDefaultAiConfig(theme);

    return config != null
        ? MessageBubbleConfig(
            backgroundColor:
                config!.backgroundColor ?? defaultConfig.backgroundColor,
            gradient: config!.gradient ?? defaultConfig.gradient,
            borderColor: config!.borderColor ?? defaultConfig.borderColor,
            borderRadius: config!.borderRadius,
            padding: config!.padding,
            margin: config!.margin,
            shadows: config!.shadows ?? defaultConfig.shadows,
            maxWidth: config!.maxWidth,
            showAvatar: config!.showAvatar,
            avatar: config!.avatar ?? defaultConfig.avatar,
            showTimestamp: config!.showTimestamp,
            showStatus: config!.showStatus,
            textStyle: config!.textStyle ?? defaultConfig.textStyle,
          )
        : defaultConfig;
  }

  MessageBubbleConfig _getDefaultUserConfig(ThemeData theme) {
    return MessageBubbleConfig(
      gradient: const LinearGradient(
        colors: [AppTheme.primaryBlue, AppTheme.secondaryPurple],
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
      ),
      shadows: [
        BoxShadow(
          color: AppTheme.primaryBlue.withValues(alpha: 0.1),
          blurRadius: 12,
          offset: const Offset(0, 4),
        ),
      ],
      textStyle: const TextStyle(
        color: Colors.white,
        fontSize: 14.5,
        height: 1.5,
        fontWeight: FontWeight.w400,
      ),
      showAvatar: false,
    );
  }

  MessageBubbleConfig _getDefaultAiConfig(ThemeData theme) {
    final isDark = theme.brightness == Brightness.dark;

    return MessageBubbleConfig(
      backgroundColor: isDark
          ? const Color(0xFF374151).withValues(alpha: 0.8)
          : Colors.white.withValues(alpha: 0.9),
      borderColor: isDark
          ? const Color(0xFF4B5563).withValues(alpha: 0.5)
          : const Color(0xFFE5E7EB).withValues(alpha: 0.8),
      shadows: [
        BoxShadow(
          color: Colors.black.withValues(alpha: 0.1),
          blurRadius: 12,
          offset: const Offset(0, 4),
        ),
      ],
      textStyle: TextStyle(
        color: isDark
            ? Colors.white.withValues(alpha: 0.95)
            : const Color(0xFF111827),
        fontSize: 14.5,
        height: 1.5,
        fontWeight: FontWeight.w400,
      ),
      avatar: const _Avatar(),
    );
  }

  void _copyToClipboard(BuildContext context, String text) {
    Clipboard.setData(ClipboardData(text: text));

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Message copied to clipboard'),
        duration: Duration(seconds: 2),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }
}

class MessageBubbleConfig {
  final Color? backgroundColor;
  final Gradient? gradient;
  final Color? borderColor;
  final double borderRadius;
  final EdgeInsetsGeometry padding;
  final EdgeInsetsGeometry margin;
  final List<BoxShadow>? shadows;
  final double maxWidth;
  final bool showAvatar;
  final Widget? avatar;
  final bool showTimestamp;
  final bool showStatus;
  final TextStyle? textStyle;

  const MessageBubbleConfig({
    this.backgroundColor,
    this.gradient,
    this.borderColor,
    this.borderRadius = 22,
    this.padding = const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
    this.margin = const EdgeInsets.symmetric(vertical: 4),
    this.shadows,
    this.maxWidth = 0.78,
    this.showAvatar = true,
    this.avatar,
    this.showTimestamp = true,
    this.showStatus = false,
    this.textStyle,
  });

  MessageBubbleConfig copyWith({
    Color? backgroundColor,
    Gradient? gradient,
    Color? borderColor,
    double? borderRadius,
    EdgeInsetsGeometry? padding,
    EdgeInsetsGeometry? margin,
    List<BoxShadow>? shadows,
    double? maxWidth,
    bool? showAvatar,
    Widget? avatar,
    bool? showTimestamp,
    bool? showStatus,
    TextStyle? textStyle,
  }) {
    return MessageBubbleConfig(
      backgroundColor: backgroundColor ?? this.backgroundColor,
      gradient: gradient ?? this.gradient,
      borderColor: borderColor ?? this.borderColor,
      borderRadius: borderRadius ?? this.borderRadius,
      padding: padding ?? this.padding,
      margin: margin ?? this.margin,
      shadows: shadows ?? this.shadows,
      maxWidth: maxWidth ?? this.maxWidth,
      showAvatar: showAvatar ?? this.showAvatar,
      avatar: avatar ?? this.avatar,
      showTimestamp: showTimestamp ?? this.showTimestamp,
      showStatus: showStatus ?? this.showStatus,
      textStyle: textStyle ?? this.textStyle,
    );
  }
}

/// Message avatar component
class _Avatar extends StatelessWidget {
  final Widget? avatar;

  const _Avatar({this.avatar});

  @override
  Widget build(BuildContext context) {
    return avatar ?? _default();
  }

  Widget _default() {
    return Container(
      width: 28,
      height: 28,
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [AppTheme.primaryBlue, AppTheme.secondaryPurple],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(14),
        boxShadow: [
          BoxShadow(
            color: AppTheme.primaryBlue.withValues(alpha: 0.3),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: const Icon(Icons.fitness_center, color: Colors.white, size: 14),
    );
  }
}

/// Message sender info component
class _Sender extends StatelessWidget {
  final Widget? avatar;
  final String? senderName;
  final MessageBubbleConfig config;

  const _Sender({
    this.avatar,
    this.senderName,
    required this.config,
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(left: 8, bottom: 6),
      child: Row(
        children: [
          avatar ?? config.avatar ?? const _Avatar(),
          const SizedBox(width: 10),
          Text(
            senderName ?? 'Fitvise AI',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: Theme.of(
                context,
              ).textTheme.bodySmall?.color?.withValues(alpha: 0.7),
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }
}

/// Message status indicator component
class _StatusIndicator extends StatelessWidget {
  final String? status;

  const _StatusIndicator({this.status, super.key});

  @override
  Widget build(BuildContext context) {
    if (status == null) return const SizedBox.shrink();

    IconData icon;
    Color color;

    switch (status) {
      case 'sent':
        icon = Icons.check;
        color = Colors.grey;
        break;
      case 'delivered':
        icon = Icons.done_all;
        color = Colors.grey;
        break;
      case 'read':
        icon = Icons.done_all;
        color = AppTheme.primaryBlue;
        break;
      default:
        icon = Icons.schedule;
        color = Colors.grey;
    }

    return Icon(icon, size: 12, color: color);
  }
}

/// Message timestamp component
class _Time extends StatelessWidget {
  final DateTime timestamp;
  final bool isUser;

  const _Time({required this.timestamp, required this.isUser, super.key});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.only(
        left: isUser ? 0 : 38,
        right: isUser ? 8 : 0,
        top: 6,
      ),
      child: Text(
        _formatTime(timestamp),
        style: Theme.of(context).textTheme.bodySmall?.copyWith(
          color: Theme.of(
            context,
          ).textTheme.bodySmall?.color?.withValues(alpha: 0.5),
          fontSize: 11,
          fontWeight: FontWeight.w400,
        ),
      ),
    );
  }

  String _formatTime(DateTime timestamp) {
    final hour = timestamp.hour > 12
        ? timestamp.hour - 12
        : (timestamp.hour == 0 ? 12 : timestamp.hour);
    final period = timestamp.hour >= 12 ? 'PM' : 'AM';
    final minute = timestamp.minute.toString().padLeft(2, '0');
    return '$hour:$minute $period';
  }
}

/// Message content component
class MessageContent extends ConsumerWidget {
  final String messageId;
  final bool isUser;
  final MessageBubbleConfig config;
  final bool isEdited;
  final List<Widget>? messageActions;
  final String? status;

  const MessageContent({
    required this.messageId,
    required this.isUser,
    required this.config,
    required this.isEdited,
    this.messageActions,
    this.status,
    super.key,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final messageText = ref.watch(
      messageProvider(messageId).select((s) => s.text),
    );

    return Row(
      mainAxisAlignment: isUser
          ? MainAxisAlignment.end
          : MainAxisAlignment.start,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Flexible(
          child: Container(
            constraints: BoxConstraints(
              maxWidth: MediaQuery.of(context).size.width * config.maxWidth,
            ),
            child: Stack(
              children: [
                Container(
                  padding: config.padding,
                  margin: EdgeInsets.only(left: isUser ? 0 : 10),
                  decoration: BoxDecoration(
                    gradient: config.gradient,
                    color: config.backgroundColor,
                    borderRadius: BorderRadius.circular(config.borderRadius),
                    border: config.borderColor != null
                        ? Border.all(color: config.borderColor!)
                        : null,
                    boxShadow: config.shadows,
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(messageText, style: config.textStyle),
                      if (isEdited)
                        Padding(
                          padding: const EdgeInsets.only(top: 6),
                          child: Text(
                            'edited',
                            style: TextStyle(
                              color: isUser
                                  ? Colors.white.withValues(alpha: 0.7)
                                  : Theme.of(context).textTheme.bodySmall?.color
                                        ?.withValues(alpha: 0.5),
                              fontSize: 11,
                              fontStyle: FontStyle.italic,
                              fontWeight: FontWeight.w300,
                            ),
                          ),
                        ),
                      if (config.showStatus && status != null)
                        Padding(
                          padding: const EdgeInsets.only(top: 4),
                          child: _StatusIndicator(status: status),
                        ),
                    ],
                  ),
                ),
                if (messageActions != null && messageActions!.isNotEmpty)
                  Positioned(
                    right: isUser ? -10 : null,
                    left: !isUser ? -10 : null,
                    top: 0,
                    bottom: 0,
                    child: SizedBox(
                      width: 90,
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: messageActions!,
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

/// Message actions component
class _ActionButtonContainer extends StatelessWidget {
  final List<Widget> actions;
  final bool isUser;

  const _ActionButtonContainer({
    required this.actions,
    required this.isUser,
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.only(left: isUser ? 0 : 38, top: 12),
      child: Wrap(spacing: 8, runSpacing: 8, children: actions),
    );
  }
}

class _ActionButton extends StatelessWidget {
  final MessageAction action;
  final void Function(MessageAction)? onTap;

  const _ActionButton({required this.action, this.onTap});

  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 200),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(24),
          onTap: () => onTap?.call(action),
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [
                  AppTheme.primaryBlue.withValues(alpha: 0.1),
                  AppTheme.secondaryPurple.withValues(alpha: 0.1),
                ],
              ),
              borderRadius: BorderRadius.circular(24),
              border: Border.all(
                color: AppTheme.primaryBlue.withValues(alpha: 0.3),
              ),
            ),
            child: Text(
              action.label,
              style: TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w500,
                color: AppTheme.primaryBlue,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// message actions (copy, edit, etc.) component
class ContentAction extends StatelessWidget {
  final IconData icon;
  final VoidCallback? onPressed;

  const ContentAction({required this.icon, this.onPressed, super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 2),
      child: Material(
        color: Theme.of(context).brightness == Brightness.dark
            ? const Color(0xFF374151).withValues(alpha: 0.9)
            : Colors.white.withValues(alpha: 0.95),
        borderRadius: BorderRadius.circular(18),
        elevation: 2,
        shadowColor: Colors.black.withValues(alpha: 0.1),
        child: InkWell(
          borderRadius: BorderRadius.circular(18),
          onTap: onPressed,
          child: Container(
            padding: const EdgeInsets.all(8),
            child: Icon(
              icon,
              size: 16,
              color: Theme.of(context).brightness == Brightness.dark
                  ? Colors.white.withValues(alpha: 0.8)
                  : const Color(0xFF374151).withValues(alpha: 0.8),
            ),
          ),
        ),
      ),
    );
  }
}
