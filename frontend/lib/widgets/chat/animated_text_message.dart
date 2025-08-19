import 'package:flutter/material.dart';

/// A widget that animates text word-by-word for streaming chat messages
/// 
/// Features:
/// - Word-by-word streaming animation
/// - Customizable animation timing
/// - Support for different text styles
/// - Smooth fade and slide animations
/// - Performance optimized for long texts
class AnimatedTextMessage extends StatefulWidget {
  /// The text to animate
  final String text;
  
  /// Style for the animated text
  final TextStyle? style;
  
  /// Duration between each word animation
  final Duration wordDelay;
  
  /// Duration for each word's fade animation
  final Duration wordDuration;
  
  /// Whether the animation should start automatically
  final bool autoStart;
  
  /// Callback when animation completes
  final VoidCallback? onComplete;
  
  /// Whether to show a typing cursor at the end
  final bool showCursor;
  
  /// Custom cursor widget
  final Widget? cursor;

  const AnimatedTextMessage({
    super.key,
    required this.text,
    this.style,
    this.wordDelay = const Duration(milliseconds: 100),
    this.wordDuration = const Duration(milliseconds: 300),
    this.autoStart = true,
    this.onComplete,
    this.showCursor = false,
    this.cursor,
  });

  @override
  State<AnimatedTextMessage> createState() => _AnimatedTextMessageState();
}

class _AnimatedTextMessageState extends State<AnimatedTextMessage>
    with TickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;
  late List<String> _words;
  int _currentWordIndex = 0;
  bool _isAnimating = false;

  @override
  void initState() {
    super.initState();
    
    _words = widget.text.split(' ');
    _animationController = AnimationController(
      duration: widget.wordDuration,
      vsync: this,
    );
    
    _fadeAnimation = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeOut,
    ));

    if (widget.autoStart) {
      _startAnimation();
    }
  }

  @override
  void didUpdateWidget(AnimatedTextMessage oldWidget) {
    super.didUpdateWidget(oldWidget);
    
    if (oldWidget.text != widget.text) {
      _words = widget.text.split(' ');
      _currentWordIndex = 0;
      if (widget.autoStart) {
        _startAnimation();
      }
    }
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  void _startAnimation() {
    if (_isAnimating) return;
    
    setState(() {
      _isAnimating = true;
      _currentWordIndex = 0;
    });
    
    _animateNextWord();
  }

  void _animateNextWord() {
    if (_currentWordIndex < _words.length) {
      _animationController.forward().then((_) {
        setState(() {
          _currentWordIndex++;
        });
        
        if (_currentWordIndex < _words.length) {
          _animationController.reset();
          Future.delayed(widget.wordDelay, _animateNextWord);
        } else {
          _isAnimating = false;
          widget.onComplete?.call();
        }
      });
    }
  }

  /// Manually start the animation
  void startAnimation() {
    _startAnimation();
  }

  /// Skip to the end of the animation
  void skipAnimation() {
    setState(() {
      _currentWordIndex = _words.length;
      _isAnimating = false;
    });
    _animationController.stop();
    widget.onComplete?.call();
  }

  @override
  Widget build(BuildContext context) {
    if (widget.text.isEmpty) {
      return const SizedBox.shrink();
    }

    return Wrap(
      children: [
        // Already animated words
        ..._words.take(_currentWordIndex).map((word) {
          return Text(
            '$word ',
            style: widget.style,
          );
        }),
        
        // Currently animating word
        if (_currentWordIndex < _words.length)
          AnimatedBuilder(
            animation: _fadeAnimation,
            builder: (context, child) {
              return Opacity(
                opacity: _fadeAnimation.value,
                child: Transform.translate(
                  offset: Offset(0, 5 * (1 - _fadeAnimation.value)),
                  child: Text(
                    '${_words[_currentWordIndex]} ',
                    style: widget.style,
                  ),
                ),
              );
            },
          ),
        
        // Typing cursor
        if (widget.showCursor && _isAnimating)
          widget.cursor ?? _buildDefaultCursor(),
      ],
    );
  }

  Widget _buildDefaultCursor() {
    return AnimatedBuilder(
      animation: _animationController,
      builder: (context, child) {
        return Opacity(
          opacity: _animationController.value,
          child: Container(
            width: 2,
            height: 16,
            color: widget.style?.color ?? Colors.black,
            margin: const EdgeInsets.only(left: 2),
          ),
        );
      },
    );
  }
}

/// A simplified version for static text with fade-in animation
class FadeInText extends StatefulWidget {
  final String text;
  final TextStyle? style;
  final Duration duration;
  final bool autoStart;

  const FadeInText({
    super.key,
    required this.text,
    this.style,
    this.duration = const Duration(milliseconds: 500),
    this.autoStart = true,
  });

  @override
  State<FadeInText> createState() => _FadeInTextState();
}

class _FadeInTextState extends State<FadeInText>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();
    
    _controller = AnimationController(
      duration: widget.duration,
      vsync: this,
    );
    
    _fadeAnimation = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOut,
    ));

    if (widget.autoStart) {
      _controller.forward();
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _fadeAnimation,
      builder: (context, child) {
        return Opacity(
          opacity: _fadeAnimation.value,
          child: Transform.translate(
            offset: Offset(0, 10 * (1 - _fadeAnimation.value)),
            child: Text(
              widget.text,
              style: widget.style,
            ),
          ),
        );
      },
    );
  }
}