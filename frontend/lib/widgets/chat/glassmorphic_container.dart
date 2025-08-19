import 'dart:ui';
import 'package:flutter/material.dart';

/// Configuration for glassmorphic container appearance
class GlassmorphicConfig {
  final double blur;
  final double opacity;
  final Color? color;
  final Gradient? gradient;
  final Border? border;
  final BorderRadiusGeometry borderRadius;
  final List<BoxShadow>? shadows;
  final EdgeInsetsGeometry? padding;
  final EdgeInsetsGeometry? margin;

  const GlassmorphicConfig({
    this.blur = 10.0,
    this.opacity = 0.1,
    this.color,
    this.gradient,
    this.border,
    this.borderRadius = const BorderRadius.all(Radius.circular(16)),
    this.shadows,
    this.padding,
    this.margin,
  });

  GlassmorphicConfig copyWith({
    double? blur,
    double? opacity,
    Color? color,
    Gradient? gradient,
    Border? border,
    BorderRadiusGeometry? borderRadius,
    List<BoxShadow>? shadows,
    EdgeInsetsGeometry? padding,
    EdgeInsetsGeometry? margin,
  }) {
    return GlassmorphicConfig(
      blur: blur ?? this.blur,
      opacity: opacity ?? this.opacity,
      color: color ?? this.color,
      gradient: gradient ?? this.gradient,
      border: border ?? this.border,
      borderRadius: borderRadius ?? this.borderRadius,
      shadows: shadows ?? this.shadows,
      padding: padding ?? this.padding,
      margin: margin ?? this.margin,
    );
  }
}

/// A reusable glassmorphic container widget with blur effects
/// 
/// Features:
/// - Customizable blur and opacity
/// - Gradient support
/// - Border customization
/// - Shadow effects
/// - Responsive design
/// - Performance optimized
/// - Multiple glass styles
class GlassmorphicContainer extends StatelessWidget {
  /// The child widget to display inside the container
  final Widget child;
  
  /// Configuration for the glassmorphic effect
  final GlassmorphicConfig config;
  
  /// Width of the container
  final double? width;
  
  /// Height of the container
  final double? height;
  
  /// Alignment of the child widget
  final AlignmentGeometry? alignment;
  
  /// Whether to clip the child widget to the container bounds
  final bool clipBehavior;

  const GlassmorphicContainer({
    super.key,
    required this.child,
    required this.config,
    this.width,
    this.height,
    this.alignment,
    this.clipBehavior = true,
  });

  /// Creates a light glassmorphic container
  factory GlassmorphicContainer.light({
    Key? key,
    required Widget child,
    double? width,
    double? height,
    EdgeInsetsGeometry? padding,
    EdgeInsetsGeometry? margin,
    BorderRadiusGeometry? borderRadius,
    double blur = 15.0,
    double opacity = 0.1,
  }) {
    return GlassmorphicContainer(
      key: key,
      width: width,
      height: height,
      config: GlassmorphicConfig(
        blur: blur,
        opacity: opacity,
        color: Colors.white,
        border: Border.all(
          color: Colors.white.withValues(alpha: 0.2),
          width: 1,
        ),
        borderRadius: borderRadius ?? const BorderRadius.all(Radius.circular(16)),
        shadows: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.1),
            blurRadius: 20,
            offset: const Offset(0, 8),
          ),
        ],
        padding: padding,
        margin: margin,
      ),
      child: child,
    );
  }

  /// Creates a dark glassmorphic container
  factory GlassmorphicContainer.dark({
    Key? key,
    required Widget child,
    double? width,
    double? height,
    EdgeInsetsGeometry? padding,
    EdgeInsetsGeometry? margin,
    BorderRadiusGeometry? borderRadius,
    double blur = 15.0,
    double opacity = 0.2,
  }) {
    return GlassmorphicContainer(
      key: key,
      width: width,
      height: height,
      config: GlassmorphicConfig(
        blur: blur,
        opacity: opacity,
        color: Colors.black,
        border: Border.all(
          color: Colors.white.withValues(alpha: 0.1),
          width: 1,
        ),
        borderRadius: borderRadius ?? const BorderRadius.all(Radius.circular(16)),
        shadows: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.3),
            blurRadius: 20,
            offset: const Offset(0, 8),
          ),
        ],
        padding: padding,
        margin: margin,
      ),
      child: child,
    );
  }

  /// Creates a gradient glassmorphic container
  factory GlassmorphicContainer.gradient({
    Key? key,
    required Widget child,
    required Gradient gradient,
    double? width,
    double? height,
    EdgeInsetsGeometry? padding,
    EdgeInsetsGeometry? margin,
    BorderRadiusGeometry? borderRadius,
    double blur = 15.0,
    double opacity = 0.15,
  }) {
    return GlassmorphicContainer(
      key: key,
      width: width,
      height: height,
      config: GlassmorphicConfig(
        blur: blur,
        opacity: opacity,
        gradient: gradient,
        border: Border.all(
          color: Colors.white.withValues(alpha: 0.2),
          width: 1,
        ),
        borderRadius: borderRadius ?? const BorderRadius.all(Radius.circular(16)),
        shadows: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.15),
            blurRadius: 25,
            offset: const Offset(0, 10),
          ),
        ],
        padding: padding,
        margin: margin,
      ),
      child: child,
    );
  }

  /// Creates a frosted glass container
  factory GlassmorphicContainer.frosted({
    Key? key,
    required Widget child,
    double? width,
    double? height,
    EdgeInsetsGeometry? padding,
    EdgeInsetsGeometry? margin,
    BorderRadiusGeometry? borderRadius,
    Color? color,
  }) {
    return GlassmorphicContainer(
      key: key,
      width: width,
      height: height,
      config: GlassmorphicConfig(
        blur: 20.0,
        opacity: 0.05,
        color: color ?? Colors.white,
        border: Border.all(
          color: Colors.white.withValues(alpha: 0.3),
          width: 1.5,
        ),
        borderRadius: borderRadius ?? const BorderRadius.all(Radius.circular(20)),
        shadows: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.08),
            blurRadius: 30,
            offset: const Offset(0, 15),
          ),
          BoxShadow(
            color: Colors.white.withValues(alpha: 0.1),
            blurRadius: 5,
            offset: const Offset(0, -2),
          ),
        ],
        padding: padding,
        margin: margin,
      ),
      child: child,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      width: width,
      height: height,
      margin: config.margin,
      alignment: alignment,
      child: ClipRRect(
        borderRadius: config.borderRadius,
        clipBehavior: clipBehavior ? Clip.antiAlias : Clip.none,
        child: BackdropFilter(
          filter: ImageFilter.blur(
            sigmaX: config.blur,
            sigmaY: config.blur,
          ),
          child: Container(
            padding: config.padding,
            decoration: BoxDecoration(
              color: config.color?.withValues(alpha: config.opacity),
              gradient: config.gradient != null
                  ? _createOpacityGradient(config.gradient!, config.opacity)
                  : null,
              border: config.border,
              borderRadius: config.borderRadius,
              boxShadow: config.shadows,
            ),
            child: child,
          ),
        ),
      ),
    );
  }

  Gradient _createOpacityGradient(Gradient gradient, double opacity) {
    if (gradient is LinearGradient) {
      return LinearGradient(
        begin: gradient.begin,
        end: gradient.end,
        colors: gradient.colors.map((color) => color.withValues(alpha: opacity)).toList(),
        stops: gradient.stops,
        transform: gradient.transform,
        tileMode: gradient.tileMode,
      );
    } else if (gradient is RadialGradient) {
      return RadialGradient(
        center: gradient.center,
        radius: gradient.radius,
        colors: gradient.colors.map((color) => color.withValues(alpha: opacity)).toList(),
        stops: gradient.stops,
        transform: gradient.transform,
        tileMode: gradient.tileMode,
        focal: gradient.focal,
        focalRadius: gradient.focalRadius,
      );
    } else if (gradient is SweepGradient) {
      return SweepGradient(
        center: gradient.center,
        startAngle: gradient.startAngle,
        endAngle: gradient.endAngle,
        colors: gradient.colors.map((color) => color.withValues(alpha: opacity)).toList(),
        stops: gradient.stops,
        transform: gradient.transform,
        tileMode: gradient.tileMode,
      );
    }
    
    return gradient;
  }
}

/// A specialized glassmorphic card for common use cases
class GlassmorphicCard extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry? padding;
  final EdgeInsetsGeometry? margin;
  final VoidCallback? onTap;
  final Color? color;
  final double elevation;

  const GlassmorphicCard({
    super.key,
    required this.child,
    this.padding,
    this.margin,
    this.onTap,
    this.color,
    this.elevation = 2,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;
    
    Widget card = GlassmorphicContainer.frosted(
      padding: padding ?? const EdgeInsets.all(16),
      margin: margin,
      color: color ?? (isDark ? Colors.white : Colors.white),
      child: child,
    );

    if (onTap != null) {
      card = Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: const BorderRadius.all(Radius.circular(20)),
          onTap: onTap,
          child: card,
        ),
      );
    }

    return card;
  }
}

/// A glassmorphic modal or dialog
class GlassmorphicModal extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry? padding;
  final EdgeInsetsGeometry? margin;
  final bool dismissible;
  final VoidCallback? onDismiss;

  const GlassmorphicModal({
    super.key,
    required this.child,
    this.padding,
    this.margin,
    this.dismissible = true,
    this.onDismiss,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: dismissible ? (onDismiss ?? () => Navigator.of(context).pop()) : null,
      child: Container(
        color: Colors.black.withValues(alpha: 0.5),
        child: Center(
          child: GestureDetector(
            onTap: () {}, // Prevent dismissal when tapping the modal content
            child: GlassmorphicContainer.frosted(
              width: MediaQuery.of(context).size.width * 0.9,
              padding: padding ?? const EdgeInsets.all(24),
              margin: margin ?? const EdgeInsets.all(20),
              borderRadius: const BorderRadius.all(Radius.circular(24)),
              child: child,
            ),
          ),
        ),
      ),
    );
  }
}

/// Glassmorphic floating action button
class GlassmorphicFAB extends StatelessWidget {
  final Widget child;
  final VoidCallback? onPressed;
  final Color? backgroundColor;
  final double size;

  const GlassmorphicFAB({
    super.key,
    required this.child,
    this.onPressed,
    this.backgroundColor,
    this.size = 56,
  });

  @override
  Widget build(BuildContext context) {
    return GlassmorphicContainer.gradient(
      width: size,
      height: size,
      borderRadius: BorderRadius.circular(size / 2),
      gradient: LinearGradient(
        colors: [
          (backgroundColor ?? Theme.of(context).primaryColor).withValues(alpha: 0.8),
          (backgroundColor ?? Theme.of(context).primaryColor).withValues(alpha: 0.6),
        ],
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(size / 2),
          onTap: onPressed,
          child: Center(child: child),
        ),
      ),
    );
  }
}

/// Extension methods for easy glassmorphic effects
extension GlassmorphicExtension on Widget {
  /// Wraps the widget in a glassmorphic container
  Widget glassmorphic({
    GlassmorphicConfig? config,
    double? width,
    double? height,
  }) {
    return GlassmorphicContainer(
      config: config ?? const GlassmorphicConfig(),
      width: width,
      height: height,
      child: this,
    );
  }

  /// Wraps the widget in a light glassmorphic container
  Widget glassLight({
    EdgeInsetsGeometry? padding,
    EdgeInsetsGeometry? margin,
    BorderRadiusGeometry? borderRadius,
  }) {
    return GlassmorphicContainer.light(
      padding: padding,
      margin: margin,
      borderRadius: borderRadius,
      child: this,
    );
  }

  /// Wraps the widget in a dark glassmorphic container
  Widget glassDark({
    EdgeInsetsGeometry? padding,
    EdgeInsetsGeometry? margin,
    BorderRadiusGeometry? borderRadius,
  }) {
    return GlassmorphicContainer.dark(
      padding: padding,
      margin: margin,
      borderRadius: borderRadius,
      child: this,
    );
  }

  /// Wraps the widget in a frosted glassmorphic container
  Widget glassFrosted({
    EdgeInsetsGeometry? padding,
    EdgeInsetsGeometry? margin,
    BorderRadiusGeometry? borderRadius,
    Color? color,
  }) {
    return GlassmorphicContainer.frosted(
      padding: padding,
      margin: margin,
      borderRadius: borderRadius,
      color: color,
      child: this,
    );
  }
}