import 'package:flutter/material.dart';
import '../../theme/app_theme.dart';

/// Configuration class for chat input appearance and behavior
class ChatInputConfig {
  final String? hintText;
  final int? maxLines;
  final int? maxLength;
  final TextStyle? textStyle;
  final TextStyle? hintStyle;
  final EdgeInsetsGeometry padding;
  final EdgeInsetsGeometry margin;
  final double borderRadius;
  final Color? backgroundColor;
  final Gradient? gradient;
  final Color? borderColor;
  final List<BoxShadow>? shadows;
  final bool showCharacterCount;
  final bool showStatusText;
  final bool enableVoiceInput;
  final bool enableAttachments;
  final List<AttachmentOption> attachmentOptions;

  const ChatInputConfig({
    this.hintText,
    this.maxLines,
    this.maxLength = 2000,
    this.textStyle,
    this.hintStyle,
    this.padding = const EdgeInsets.all(20),
    this.margin = EdgeInsets.zero,
    this.borderRadius = 28,
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
}

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
class ChatInput extends StatefulWidget {
  /// Text editing controller
  final TextEditingController controller;
  
  /// Configuration for input appearance
  final ChatInputConfig? config;
  
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
  
  /// Custom attachment options
  final List<AttachmentOption>? attachmentOptions;

  const ChatInput({
    super.key,
    required this.controller,
    this.config,
    this.onSend,
    this.onChanged,
    this.onVoiceToggle,
    this.isRecording = false,
    this.isLoading = false,
    this.statusText,
    this.enabled = true,
    this.focusNode,
    this.attachmentOptions,
  });

  @override
  State<ChatInput> createState() => _ChatInputState();
}

class _ChatInputState extends State<ChatInput> with TickerProviderStateMixin {
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
    
    _sendButtonScale = Tween<double>(
      begin: 0.8,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _sendButtonController,
      curve: Curves.elasticOut,
    ));
    
    _recordingPulse = Tween<double>(
      begin: 0.8,
      end: 1.2,
    ).animate(CurvedAnimation(
      parent: _recordingController,
      curve: Curves.easeInOut,
    ));

    widget.controller.addListener(_onTextChanged);
    _updateSendButton();
  }

  @override
  void didUpdateWidget(ChatInput oldWidget) {
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
    final config = widget.config ?? _getDefaultConfig(theme);
    
    return Container(
      padding: config.padding,
      margin: config.margin,
      decoration: BoxDecoration(
        color: theme.scaffoldBackgroundColor.withValues(alpha: 0.95),
        border: Border(
          top: BorderSide(
            color: theme.dividerColor.withValues(alpha: 0.3),
          ),
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
          if (config.enableAttachments)
            _buildAttachmentOptions(config),

          // Main input area
          _buildInputArea(context, config),

          // Status and character count
          if (config.showStatusText || config.showCharacterCount)
            _buildStatusArea(context, config),
        ],
      ),
    );
  }

  ChatInputConfig _getDefaultConfig(ThemeData theme) {
    return ChatInputConfig(
      hintText: 'Ask about workouts, nutrition, or fitness goals...',
      attachmentOptions: [
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
      ],
    );
  }

  Widget _buildAttachmentOptions(ChatInputConfig config) {
    final options = widget.attachmentOptions ?? config.attachmentOptions;
    
    if (options.isEmpty) return const SizedBox.shrink();
    
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      child: Row(
        children: options.map((option) {
          return Container(
            margin: const EdgeInsets.only(right: 10),
            child: _buildAttachmentButton(
              icon: option.icon,
              tooltip: option.tooltip,
              onPressed: option.onPressed ?? () => _handleAttachment(option.type),
            ),
          );
        }).toList(),
      ),
    );
  }

  Widget _buildAttachmentButton({
    required IconData icon,
    required String tooltip,
    required VoidCallback onPressed,
  }) {
    final theme = Theme.of(context);
    
    return Tooltip(
      message: tooltip,
      child: Container(
        decoration: BoxDecoration(
          color: theme.brightness == Brightness.dark
              ? const Color(0xFF374151).withValues(alpha: 0.5)
              : Colors.white.withValues(alpha: 0.8),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: theme.dividerColor.withValues(alpha: 0.3),
          ),
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
                size: 20,
                color: theme.iconTheme.color?.withValues(alpha: 0.7),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildInputArea(BuildContext context, ChatInputConfig config) {
    final theme = Theme.of(context);
    
    return Container(
      padding: const EdgeInsets.all(4),
      decoration: BoxDecoration(
        gradient: config.gradient ?? LinearGradient(
          colors: [
            AppTheme.primaryBlue.withValues(alpha: 0.1),
            AppTheme.secondaryPurple.withValues(alpha: 0.1),
          ],
        ),
        borderRadius: BorderRadius.circular(config.borderRadius),
        border: Border.all(
          color: config.borderColor ?? AppTheme.primaryBlue.withValues(alpha: 0.2),
          width: 1.5,
        ),
        boxShadow: config.shadows,
      ),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        decoration: BoxDecoration(
          color: config.backgroundColor ?? (theme.brightness == Brightness.dark
              ? const Color(0xFF374151).withValues(alpha: 0.9)
              : Colors.white.withValues(alpha: 0.95)),
          borderRadius: BorderRadius.circular(config.borderRadius - 4),
        ),
        child: Row(
          children: [
            // Text input
            Expanded(
              child: TextField(
                controller: widget.controller,
                focusNode: widget.focusNode,
                maxLines: config.maxLines,
                maxLength: config.maxLength,
                enabled: widget.enabled,
                textCapitalization: TextCapitalization.sentences,
                textInputAction: TextInputAction.send,
                style: config.textStyle ?? TextStyle(
                  color: theme.brightness == Brightness.dark
                      ? Colors.white
                      : const Color(0xFF111827),
                  fontSize: 15,
                  height: 1.4,
                ),
                decoration: InputDecoration(
                  hintText: config.hintText,
                  hintStyle: config.hintStyle ?? TextStyle(
                    color: theme.textTheme.bodySmall?.color?.withValues(alpha: 0.6),
                    fontSize: 15,
                  ),
                  border: InputBorder.none,
                  counterText: '', // Hide default counter
                  contentPadding: EdgeInsets.zero,
                ),
                onSubmitted: widget.enabled ? (_) => _handleSend() : null,
              ),
            ),
            const SizedBox(width: 12),

            // Voice recording button
            if (config.enableVoiceInput)
              _buildVoiceButton(),
            
            if (config.enableVoiceInput)
              const SizedBox(width: 8),

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
                    widget.isRecording ? Icons.mic_off_rounded : Icons.mic_rounded,
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
                            valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                          ),
                        )
                      : Icon(
                          Icons.send_rounded,
                          size: 16,
                          color: canSend ? Colors.white : Theme.of(context).disabledColor,
                        ),
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildStatusArea(BuildContext context, ChatInputConfig config) {
    final theme = Theme.of(context);
    final statusText = widget.statusText ?? _getDefaultStatusText();
    final characterCount = widget.controller.text.length;
    final maxLength = config.maxLength ?? 2000;
    
    return Padding(
      padding: const EdgeInsets.only(top: 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          // Status text
          if (config.showStatusText)
            Expanded(
              child: Text(
                statusText,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.textTheme.bodySmall?.color?.withValues(alpha: 0.6),
                  fontSize: 12,
                ),
              ),
            ),

          // Character count
          if (config.showCharacterCount)
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

/// A simplified chat input for basic use cases
class SimpleChatInput extends StatelessWidget {
  final TextEditingController controller;
  final ValueChanged<String>? onSend;
  final String? hintText;
  final bool enabled;

  const SimpleChatInput({
    super.key,
    required this.controller,
    this.onSend,
    this.hintText,
    this.enabled = true,
  });

  @override
  Widget build(BuildContext context) {
    return ChatInput(
      controller: controller,
      onSend: onSend,
      enabled: enabled,
      config: ChatInputConfig(
        hintText: hintText ?? 'Type a message...',
        enableVoiceInput: false,
        enableAttachments: false,
        showStatusText: false,
        showCharacterCount: false,
      ),
    );
  }
}