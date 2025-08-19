import 'package:flutter/material.dart';
import '../../theme/app_theme.dart';

/// Configuration for loading widget appearance
class LoadingConfig {
  final Color? color;
  final Gradient? gradient;
  final double size;
  final Duration duration;
  final String? text;
  final TextStyle? textStyle;
  final EdgeInsetsGeometry padding;
  final EdgeInsetsGeometry margin;

  const LoadingConfig({
    this.color,
    this.gradient,
    this.size = 10.0,
    this.duration = const Duration(milliseconds: 1500),
    this.text,
    this.textStyle,
    this.padding = const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
    this.margin = const EdgeInsets.only(left: 10),
  });
}

/// A collection of loading widgets for chat interfaces
/// 
/// Features:
/// - Multiple loading animation styles
/// - Customizable appearance
/// - Typing indicators
/// - Progress indicators
/// - Voice recording indicators
/// - Smooth animations
class ChatLoadingWidget extends StatefulWidget {
  /// Type of loading animation
  final LoadingType type;
  
  /// Configuration for appearance
  final LoadingConfig? config;
  
  /// Whether the loading animation is active
  final bool isActive;
  
  /// Custom message to display
  final String? message;

  const ChatLoadingWidget({
    super.key,
    this.type = LoadingType.typing,
    this.config,
    this.isActive = true,
    this.message,
  });

  @override
  State<ChatLoadingWidget> createState() => _ChatLoadingWidgetState();
}

class _ChatLoadingWidgetState extends State<ChatLoadingWidget>
    with TickerProviderStateMixin {
  late AnimationController _controller;
  late List<AnimationController> _dotControllers;

  @override
  void initState() {
    super.initState();
    
    final config = widget.config ?? const LoadingConfig();
    _controller = AnimationController(
      duration: config.duration,
      vsync: this,
    );

    // Initialize dot controllers for typing animation
    _dotControllers = List.generate(3, (index) {
      return AnimationController(
        duration: config.duration,
        vsync: this,
      );
    });

    if (widget.isActive) {
      _startAnimation();
    }
  }

  @override
  void didUpdateWidget(ChatLoadingWidget oldWidget) {
    super.didUpdateWidget(oldWidget);
    
    if (oldWidget.isActive != widget.isActive) {
      if (widget.isActive) {
        _startAnimation();
      } else {
        _stopAnimation();
      }
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    for (final controller in _dotControllers) {
      controller.dispose();
    }
    super.dispose();
  }

  void _startAnimation() {
    switch (widget.type) {
      case LoadingType.typing:
        _startTypingAnimation();
        break;
      case LoadingType.pulse:
        _controller.repeat();
        break;
      case LoadingType.wave:
        _controller.repeat();
        break;
      case LoadingType.dots:
        _startDotsAnimation();
        break;
      case LoadingType.spinner:
        _controller.repeat();
        break;
    }
  }

  void _stopAnimation() {
    _controller.stop();
    for (final controller in _dotControllers) {
      controller.stop();
    }
  }

  void _startTypingAnimation() {
    for (int i = 0; i < _dotControllers.length; i++) {
      Future.delayed(Duration(milliseconds: i * 200), () {
        if (mounted) {
          _dotControllers[i].repeat();
        }
      });
    }
  }

  void _startDotsAnimation() {
    _controller.repeat();
  }

  @override
  Widget build(BuildContext context) {
    if (!widget.isActive) {
      return const SizedBox.shrink();
    }

    final theme = Theme.of(context);
    final config = widget.config ?? _getDefaultConfig(theme);
    
    return Container(
      alignment: Alignment.centerLeft,
      padding: const EdgeInsets.symmetric(vertical: 10),
      child: Container(
        padding: config.padding,
        margin: config.margin,
        decoration: _buildContainerDecoration(theme, config),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            _buildLoadingAnimation(config),
            if (widget.message != null) ...[
              const SizedBox(width: 12),
              Text(
                widget.message!,
                style: config.textStyle ?? _getDefaultTextStyle(theme),
              ),
            ],
          ],
        ),
      ),
    );
  }

  LoadingConfig _getDefaultConfig(ThemeData theme) {
    return LoadingConfig(
      gradient: const LinearGradient(
        colors: [AppTheme.primaryBlue, AppTheme.secondaryPurple],
      ),
      textStyle: TextStyle(
        color: theme.brightness == Brightness.dark
            ? Colors.white.withValues(alpha: 0.95)
            : const Color(0xFF111827),
        fontSize: 14,
        fontWeight: FontWeight.w500,
      ),
    );
  }

  TextStyle _getDefaultTextStyle(ThemeData theme) {
    return TextStyle(
      color: theme.brightness == Brightness.dark
          ? Colors.white.withValues(alpha: 0.95)
          : const Color(0xFF111827),
      fontSize: 14,
      fontWeight: FontWeight.w500,
    );
  }

  BoxDecoration _buildContainerDecoration(ThemeData theme, LoadingConfig config) {
    return BoxDecoration(
      color: theme.brightness == Brightness.dark
          ? const Color(0xFF374151).withValues(alpha: 0.8)
          : Colors.white.withValues(alpha: 0.9),
      borderRadius: BorderRadius.circular(22),
      border: Border.all(
        color: theme.brightness == Brightness.dark
            ? const Color(0xFF4B5563).withValues(alpha: 0.5)
            : const Color(0xFFE5E7EB).withValues(alpha: 0.8),
      ),
      boxShadow: [
        BoxShadow(
          color: Colors.black.withValues(alpha: 0.08),
          blurRadius: 12,
          offset: const Offset(0, 4),
        ),
      ],
    );
  }

  Widget _buildLoadingAnimation(LoadingConfig config) {
    switch (widget.type) {
      case LoadingType.typing:
        return _buildTypingDots(config);
      case LoadingType.pulse:
        return _buildPulseAnimation(config);
      case LoadingType.wave:
        return _buildWaveAnimation(config);
      case LoadingType.dots:
        return _buildBouncingDots(config);
      case LoadingType.spinner:
        return _buildSpinnerAnimation(config);
    }
  }

  Widget _buildTypingDots(LoadingConfig config) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: List.generate(3, (index) {
        return AnimatedBuilder(
          animation: _dotControllers[index],
          builder: (context, child) {
            final animationValue = Curves.easeInOut.transform(
              (_dotControllers[index].value + (index * 0.3)) % 1.0,
            );
            final scale = 0.7 + (0.4 * (0.5 + 0.5 * animationValue));
            final opacity = 0.3 + (0.7 * animationValue);
            
            return Padding(
              padding: const EdgeInsets.symmetric(horizontal: 3),
              child: Transform.scale(
                scale: scale,
                child: Container(
                  width: config.size,
                  height: config.size,
                  decoration: BoxDecoration(
                    gradient: config.gradient,
                    color: config.color?.withValues(alpha: opacity),
                    shape: BoxShape.circle,
                  ),
                ),
              ),
            );
          },
        );
      }),
    );
  }

  Widget _buildPulseAnimation(LoadingConfig config) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        final scale = 0.8 + (0.4 * Curves.easeInOut.transform(_controller.value));
        return Transform.scale(
          scale: scale,
          child: Container(
            width: config.size * 2,
            height: config.size * 2,
            decoration: BoxDecoration(
              gradient: config.gradient,
              color: config.color,
              shape: BoxShape.circle,
            ),
          ),
        );
      },
    );
  }

  Widget _buildWaveAnimation(LoadingConfig config) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: List.generate(5, (index) {
        return AnimatedBuilder(
          animation: _controller,
          builder: (context, child) {
            final offset = (index * 0.1);
            final animationValue = Curves.easeInOut.transform(
              (_controller.value + offset) % 1.0,
            );
            final height = config.size * (0.5 + 0.5 * animationValue);
            
            return Padding(
              padding: const EdgeInsets.symmetric(horizontal: 1),
              child: Container(
                width: 3,
                height: height,
                decoration: BoxDecoration(
                  gradient: config.gradient,
                  color: config.color,
                  borderRadius: BorderRadius.circular(1.5),
                ),
              ),
            );
          },
        );
      }),
    );
  }

  Widget _buildBouncingDots(LoadingConfig config) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: List.generate(3, (index) {
        return AnimatedBuilder(
          animation: _controller,
          builder: (context, child) {
            final offset = (index * 0.2);
            final animationValue = Curves.elasticOut.transform(
              (_controller.value + offset) % 1.0,
            );
            final translateY = -15 * animationValue;
            
            return Padding(
              padding: const EdgeInsets.symmetric(horizontal: 2),
              child: Transform.translate(
                offset: Offset(0, translateY),
                child: Container(
                  width: config.size,
                  height: config.size,
                  decoration: BoxDecoration(
                    gradient: config.gradient,
                    color: config.color,
                    shape: BoxShape.circle,
                  ),
                ),
              ),
            );
          },
        );
      }),
    );
  }

  Widget _buildSpinnerAnimation(LoadingConfig config) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return Transform.rotate(
          angle: _controller.value * 2 * 3.14159,
          child: Container(
            width: config.size * 2,
            height: config.size * 2,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              border: Border.all(
                color: Colors.transparent,
                width: 2,
              ),
              gradient: SweepGradient(
                colors: [
                  Colors.transparent,
                  config.gradient?.colors.first ?? config.color ?? AppTheme.primaryBlue,
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

/// Enum for different loading animation types
enum LoadingType {
  typing,
  pulse,
  wave,
  dots,
  spinner,
}

/// A simple typing indicator with dots
class TypingIndicator extends StatelessWidget {
  final bool isVisible;
  final String? message;
  final Color? color;

  const TypingIndicator({
    super.key,
    this.isVisible = true,
    this.message,
    this.color,
  });

  @override
  Widget build(BuildContext context) {
    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 200),
      child: isVisible
          ? ChatLoadingWidget(
              type: LoadingType.typing,
              message: message ?? 'Fitvise AI is typing...',
              config: LoadingConfig(
                color: color,
              ),
            )
          : const SizedBox.shrink(),
    );
  }
}

/// A voice recording indicator
class VoiceRecordingIndicator extends StatelessWidget {
  final bool isRecording;
  final Duration? duration;

  const VoiceRecordingIndicator({
    super.key,
    this.isRecording = false,
    this.duration,
  });

  @override
  Widget build(BuildContext context) {
    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 200),
      child: isRecording
          ? ChatLoadingWidget(
              type: LoadingType.wave,
              message: 'Recording${duration != null ? ' ${_formatDuration(duration!)}' : '...'}',
              config: const LoadingConfig(
                color: Colors.red,
                size: 12.0,
              ),
            )
          : const SizedBox.shrink(),
    );
  }

  String _formatDuration(Duration duration) {
    final minutes = duration.inMinutes;
    final seconds = duration.inSeconds % 60;
    return '${minutes.toString().padLeft(2, '0')}:${seconds.toString().padLeft(2, '0')}';
  }
}

/// A processing indicator for file uploads or operations
class ProcessingIndicator extends StatelessWidget {
  final bool isProcessing;
  final String? message;
  final double? progress;

  const ProcessingIndicator({
    super.key,
    this.isProcessing = false,
    this.message,
    this.progress,
  });

  @override
  Widget build(BuildContext context) {
    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 200),
      child: isProcessing
          ? Container(
              padding: const EdgeInsets.all(16),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (progress != null)
                    SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(
                        value: progress,
                        strokeWidth: 2,
                        valueColor: const AlwaysStoppedAnimation<Color>(AppTheme.primaryBlue),
                      ),
                    )
                  else
                    const ChatLoadingWidget(
                      type: LoadingType.spinner,
                      config: LoadingConfig(size: 8.0),
                    ),
                  const SizedBox(width: 12),
                  Text(
                    message ?? 'Processing...',
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                ],
              ),
            )
          : const SizedBox.shrink(),
    );
  }
}