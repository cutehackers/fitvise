import 'package:flutter/material.dart';
import 'package:gap/gap.dart';

import '../../theme/app_theme.dart';

/// Attachment option configuration
class AttachmentOption {
  final IconData icon;
  final String tooltip;
  final String type;
  final VoidCallback? onPressed;

  const AttachmentOption({
    required this.icon,
    required this.tooltip,
    required this.type,
    this.onPressed,
  });
}

/// A modern, feature-rich chat input widget
///
/// Features:
/// - Glassmorphic design with gradients
/// - Voice input support
/// - File attachment options
/// - Character count and status indicators
/// - Customizable appearance
/// - Auto-expanding text field
/// - Send button with loading states
/// - Keyboard shortcuts support
class MessageComposer extends StatefulWidget {
  /// Text editing controller
  final TextEditingController controller;

  /// Callback when send button is pressed
  final ValueChanged<String>? onSend;

  /// Callback when text changes
  final ValueChanged<String>? onChanged;

  /// Callback for voice input toggle
  final VoidCallback? onVoiceToggle;

  /// Whether voice recording is active
  final bool isRecording;

  /// Whether the input is in loading state
  final bool isLoading;

  /// Custom status text to display
  final String? statusText;

  /// Whether the send button is enabled
  final bool enabled;

  /// Focus node for the text field
  final FocusNode? focusNode;

  // Appearance Configuration
  /// Hint text for the input field
  final String? hintText;

  /// Maximum number of lines for the input field
  final int? maxLines;

  /// Maximum character length
  final int maxLength;

  /// Text style for input text
  final TextStyle? textStyle;

  /// Text style for hint text
  final TextStyle? hintStyle;

  /// Padding around the entire widget
  final EdgeInsetsGeometry padding;

  /// Margin around the entire widget
  final EdgeInsetsGeometry margin;

  /// Border radius for the input container
  final double borderRadius;

  /// Background color for the input container
  final Color? backgroundColor;

  /// Gradient for the input container
  final Gradient? gradient;

  /// Border color for the input container
  final Color? borderColor;

  /// Box shadows for the input container
  final List<BoxShadow>? shadows;

  /// Whether to show character count
  final bool showCharacterCount;

  /// Whether to show status text
  final bool showStatusText;

  /// Whether to enable voice input button
  final bool enableVoiceInput;

  /// Whether to enable attachment buttons
  final bool enableAttachments;

  /// Custom attachment options
  final List<AttachmentOption> attachmentOptions;

  const MessageComposer({
    super.key,
    required this.controller,
    this.onSend,
    this.onChanged,
    this.onVoiceToggle,
    this.isRecording = false,
    this.isLoading = false,
    this.statusText,
    this.enabled = true,
    this.focusNode,
    // Appearance Configuration
    this.hintText,
    this.maxLines,
    this.maxLength = 2000,
    this.textStyle,
    this.hintStyle,
    this.padding = const EdgeInsets.all(16),
    this.margin = EdgeInsets.zero,
    this.borderRadius = 8,
    this.backgroundColor,
    this.gradient,
    this.borderColor,
    this.shadows,
    this.showCharacterCount = true,
    this.showStatusText = true,
    this.enableVoiceInput = true,
    this.enableAttachments = true,
    this.attachmentOptions = const [],
  });

  @override
  State<MessageComposer> createState() => _MessageComposerState();
}

class _MessageComposerState extends State<MessageComposer>
    with TickerProviderStateMixin {
  late AnimationController _sendButtonController;
  late AnimationController _recordingController;
  late Animation<double> _sendButtonScale;
  late Animation<double> _recordingPulse;

  bool _hasText = false;

  @override
  void initState() {
    super.initState();

    _sendButtonController = AnimationController(
      duration: const Duration(milliseconds: 200),
      vsync: this,
    );

    _recordingController = AnimationController(
      duration: const Duration(milliseconds: 1000),
      vsync: this,
    );

    _sendButtonScale = Tween<double>(begin: 0.8, end: 1.0).animate(
      CurvedAnimation(parent: _sendButtonController, curve: Curves.elasticOut),
    );

    _recordingPulse = Tween<double>(begin: 0.8, end: 1.2).animate(
      CurvedAnimation(parent: _recordingController, curve: Curves.easeInOut),
    );

    widget.controller.addListener(_onTextChanged);
    _updateSendButton();
  }

  @override
  void didUpdateWidget(MessageComposer oldWidget) {
    super.didUpdateWidget(oldWidget);

    if (oldWidget.controller != widget.controller) {
      oldWidget.controller.removeListener(_onTextChanged);
      widget.controller.addListener(_onTextChanged);
      _updateSendButton();
    }

    if (oldWidget.isRecording != widget.isRecording) {
      if (widget.isRecording) {
        _recordingController.repeat(reverse: true);
      } else {
        _recordingController.stop();
        _recordingController.reset();
      }
    }
  }

  @override
  void dispose() {
    widget.controller.removeListener(_onTextChanged);
    _sendButtonController.dispose();
    _recordingController.dispose();
    super.dispose();
  }

  void _onTextChanged() {
    final hasText = widget.controller.text.trim().isNotEmpty;
    if (hasText != _hasText) {
      setState(() {
        _hasText = hasText;
      });
      _updateSendButton();
    }
    widget.onChanged?.call(widget.controller.text);
  }

  void _updateSendButton() {
    if (_hasText && widget.enabled) {
      _sendButtonController.forward();
    } else {
      _sendButtonController.reverse();
    }
  }

  void _handleSend() {
    final text = widget.controller.text.trim();
    if (text.isNotEmpty && widget.enabled && !widget.isLoading) {
      widget.onSend?.call(text);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final effectiveHintText = widget.hintText ?? _getDefaultHintText();
    final effectiveAttachmentOptions = widget.attachmentOptions.isEmpty
        ? _getDefaultAttachmentOptions()
        : widget.attachmentOptions;

    return Container(
      padding: widget.padding,
      margin: widget.margin,
      decoration: BoxDecoration(
        color: theme.scaffoldBackgroundColor.withValues(alpha: 0.95),
        border: Border(
          top: BorderSide(color: theme.dividerColor.withValues(alpha: 0.3)),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 20,
            offset: const Offset(0, -4),
          ),
        ],
      ),
      child: Column(
        children: [
          // Attachment options
          if (widget.enableAttachments)
            _buildAttachmentOptions(effectiveAttachmentOptions),

          // Main input area
          _buildInputArea(context, theme, effectiveHintText),

          // Status and character count
          if (widget.showStatusText || widget.showCharacterCount)
            _buildStatusArea(context, theme),
        ],
      ),
    );
  }

  String _getDefaultHintText() {
    return 'Ask about workouts, nutrition, or fitness goals...';
  }

  List<AttachmentOption> _getDefaultAttachmentOptions() {
    return [
      AttachmentOption(
        icon: Icons.image_rounded,
        tooltip: 'Upload workout photo',
        type: 'image',
        onPressed: () => _handleAttachment('image'),
      ),
      AttachmentOption(
        icon: Icons.attach_file_rounded,
        tooltip: 'Upload file',
        type: 'file',
        onPressed: () => _handleAttachment('file'),
      ),
      AttachmentOption(
        icon: Icons.description_rounded,
        tooltip: 'Upload fitness plan',
        type: 'document',
        onPressed: () => _handleAttachment('document'),
      ),
    ];
  }

  Widget _buildAttachmentOptions(List<AttachmentOption> options) {
    if (options.isEmpty) return const SizedBox.shrink();

    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      child: Row(
        children: options.map((option) {
          return Container(
            margin: const EdgeInsets.only(right: 10),
            child: _AttachmentButton(
              icon: option.icon,
              tooltip: option.tooltip,
              onPressed:
                  option.onPressed ?? () => _handleAttachment(option.type),
            ),
          );
        }).toList(),
      ),
    );
  }

  Widget _buildInputArea(
    BuildContext context,
    ThemeData theme,
    String hintText,
  ) {
    return Container(
      decoration: BoxDecoration(
        gradient:
            widget.gradient ??
            LinearGradient(
              colors: [
                AppTheme.primaryBlue.withValues(alpha: 0.1),
                AppTheme.secondaryPurple.withValues(alpha: 0.1),
              ],
            ),
        borderRadius: BorderRadius.circular(widget.borderRadius),
        border: Border.all(
          color:
              widget.borderColor ?? AppTheme.primaryBlue.withValues(alpha: 0.2),
          width: 1.5,
        ),
        boxShadow: widget.shadows,
      ),
      child: Container(
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color:
              widget.backgroundColor ??
              (theme.brightness == Brightness.dark
                  ? const Color(0xFF374151).withValues(alpha: 0.9)
                  : Colors.white.withValues(alpha: 0.95)),
          borderRadius: BorderRadius.circular(widget.borderRadius),
        ),
        child: Row(
          children: [
            // Text input
            Expanded(
              child: TextField(
                controller: widget.controller,
                focusNode: widget.focusNode,
                maxLines: widget.maxLines,
                maxLength: widget.maxLength,
                enabled: widget.enabled,
                textCapitalization: TextCapitalization.sentences,
                textInputAction: TextInputAction.send,
                style:
                    widget.textStyle ??
                    TextStyle(
                      color: theme.brightness == Brightness.dark
                          ? Colors.white
                          : const Color(0xFF111827),
                      fontSize: 15,
                      height: 1.4,
                    ),
                decoration: InputDecoration(
                  hintText: hintText,
                  hintStyle:
                      widget.hintStyle ??
                      TextStyle(
                        color: theme.textTheme.bodySmall?.color?.withValues(
                          alpha: 0.6,
                        ),
                        fontSize: 15,
                      ),
                  filled: true,
                  fillColor: Colors.transparent,
                  focusColor: Colors.transparent,
                  hoverColor: Colors.transparent,
                  border: InputBorder.none,
                  focusedBorder: InputBorder.none,
                  disabledBorder: InputBorder.none,
                  enabledBorder: InputBorder.none,
                  counterText: '', // Hide default counter
                  contentPadding: EdgeInsets.zero,
                ),
                onSubmitted: widget.enabled ? (_) => _handleSend() : null,
              ),
            ),
            const Gap(12),

            // Voice recording button
            if (widget.enableVoiceInput) ...[_buildVoiceButton(), const Gap(8)],

            // Send button
            _buildSendButton(),
          ],
        ),
      ),
    );
  }

  Widget _buildVoiceButton() {
    final theme = Theme.of(context);

    return AnimatedBuilder(
      animation: _recordingPulse,
      builder: (context, child) {
        return Transform.scale(
          scale: widget.isRecording ? _recordingPulse.value : 1.0,
          child: Container(
            decoration: BoxDecoration(
              color: widget.isRecording
                  ? Colors.red.withValues(alpha: 0.1)
                  : Colors.transparent,
              borderRadius: BorderRadius.circular(20),
              border: widget.isRecording
                  ? Border.all(color: Colors.red.withValues(alpha: 0.3))
                  : null,
            ),
            child: Material(
              color: Colors.transparent,
              child: InkWell(
                borderRadius: BorderRadius.circular(20),
                onTap: widget.enabled ? widget.onVoiceToggle : null,
                child: Padding(
                  padding: const EdgeInsets.all(8),
                  child: Icon(
                    widget.isRecording
                        ? Icons.mic_off_rounded
                        : Icons.mic_rounded,
                    size: 20,
                    color: widget.isRecording
                        ? Colors.red
                        : theme.iconTheme.color?.withValues(alpha: 0.7),
                  ),
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildSendButton() {
    final canSend = _hasText && widget.enabled && !widget.isLoading;

    return AnimatedBuilder(
      animation: _sendButtonScale,
      builder: (context, child) {
        return Transform.scale(
          scale: _sendButtonScale.value,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            decoration: BoxDecoration(
              gradient: canSend
                  ? const LinearGradient(
                      colors: [AppTheme.primaryBlue, AppTheme.secondaryPurple],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    )
                  : null,
              color: !canSend
                  ? Theme.of(context).disabledColor.withValues(alpha: 0.3)
                  : null,
              borderRadius: BorderRadius.circular(20),
              boxShadow: canSend
                  ? [
                      BoxShadow(
                        color: AppTheme.primaryBlue.withValues(alpha: 0.3),
                        blurRadius: 8,
                        offset: const Offset(0, 2),
                      ),
                    ]
                  : null,
            ),
            child: Material(
              color: Colors.transparent,
              child: InkWell(
                borderRadius: BorderRadius.circular(20),
                onTap: canSend ? _handleSend : null,
                child: Padding(
                  padding: const EdgeInsets.all(10),
                  child: widget.isLoading
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            valueColor: AlwaysStoppedAnimation<Color>(
                              Colors.white,
                            ),
                          ),
                        )
                      : Icon(
                          Icons.send_rounded,
                          size: 16,
                          color: canSend
                              ? Colors.white
                              : Theme.of(context).disabledColor,
                        ),
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildStatusArea(BuildContext context, ThemeData theme) {
    final statusText = widget.statusText ?? _getDefaultStatusText();
    final characterCount = widget.controller.text.length;
    final maxLength = widget.maxLength;

    return Padding(
      padding: const EdgeInsets.only(top: 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          // Status text
          if (widget.showStatusText)
            Expanded(
              child: Text(
                statusText,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.textTheme.bodySmall?.color?.withValues(
                    alpha: 0.6,
                  ),
                  fontSize: 12,
                ),
              ),
            ),

          // Character count
          if (widget.showCharacterCount)
            Text(
              '$characterCount/$maxLength',
              style: theme.textTheme.bodySmall?.copyWith(
                color: characterCount > maxLength * 0.9
                    ? Colors.red
                    : theme.textTheme.bodySmall?.color?.withValues(alpha: 0.6),
                fontSize: 12,
                fontWeight: characterCount > maxLength * 0.9
                    ? FontWeight.w600
                    : FontWeight.normal,
              ),
            ),
        ],
      ),
    );
  }

  String _getDefaultStatusText() {
    if (widget.isRecording) {
      return '🔴 Recording... Tap mic to stop';
    } else if (widget.isLoading) {
      return 'Fitvise AI is thinking...';
    }
    return 'Shift+Enter for new line';
  }

  void _handleAttachment(String type) {
    // Placeholder for file attachment handling
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('File upload ($type) coming soon!'),
        duration: const Duration(seconds: 2),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    );
  }
}

class _AttachmentButton extends StatelessWidget {
  final IconData icon;
  final String? tooltip;
  final VoidCallback? onPressed;

  const _AttachmentButton({
    required this.icon,
    this.tooltip,
    this.onPressed,
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Tooltip(
      message: tooltip,
      child: Container(
        decoration: BoxDecoration(
          color: theme.brightness == Brightness.dark
              ? const Color(0xFF374151).withValues(alpha: 0.5)
              : Colors.white.withValues(alpha: 0.8),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: theme.dividerColor.withValues(alpha: 0.3)),
        ),
        child: Material(
          color: Colors.transparent,
          child: InkWell(
            borderRadius: BorderRadius.circular(12),
            onTap: onPressed,
            child: Padding(
              padding: const EdgeInsets.all(8),
              child: Icon(
                icon,
                size: 16,
                color: theme.iconTheme.color?.withValues(alpha: 0.7),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
