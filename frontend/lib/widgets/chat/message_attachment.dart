import 'package:flutter/material.dart';
import '../../theme/app_theme.dart';

/// Configuration for attachment appearance
class AttachmentConfig {
  final double maxWidth;
  final double maxHeight;
  final EdgeInsetsGeometry padding;
  final EdgeInsetsGeometry margin;
  final BorderRadiusGeometry borderRadius;
  final Color? backgroundColor;
  final Color? borderColor;
  final List<BoxShadow>? shadows;

  const AttachmentConfig({
    this.maxWidth = 280,
    this.maxHeight = 200,
    this.padding = const EdgeInsets.all(12),
    this.margin = const EdgeInsets.symmetric(vertical: 4),
    this.borderRadius = const BorderRadius.all(Radius.circular(12)),
    this.backgroundColor,
    this.borderColor,
    this.shadows,
  });

  AttachmentConfig copyWith({
    double? maxWidth,
    double? maxHeight,
    EdgeInsetsGeometry? padding,
    EdgeInsetsGeometry? margin,
    BorderRadiusGeometry? borderRadius,
    Color? backgroundColor,
    Color? borderColor,
    List<BoxShadow>? shadows,
  }) {
    return AttachmentConfig(
      maxWidth: maxWidth ?? this.maxWidth,
      maxHeight: maxHeight ?? this.maxHeight,
      padding: padding ?? this.padding,
      margin: margin ?? this.margin,
      borderRadius: borderRadius ?? this.borderRadius,
      backgroundColor: backgroundColor ?? this.backgroundColor,
      borderColor: borderColor ?? this.borderColor,
      shadows: shadows ?? this.shadows,
    );
  }
}

/// Attachment data model
class MessageAttachment {
  final String id;
  final String name;
  final String type;
  final String? url;
  final String? localPath;
  final int? size;
  final String? mimeType;
  final Map<String, dynamic>? metadata;

  const MessageAttachment({
    required this.id,
    required this.name,
    required this.type,
    this.url,
    this.localPath,
    this.size,
    this.mimeType,
    this.metadata,
  });

  bool get isImage =>
      type == 'image' || (mimeType?.startsWith('image/') ?? false);
  bool get isVideo =>
      type == 'video' || (mimeType?.startsWith('video/') ?? false);
  bool get isAudio =>
      type == 'audio' || (mimeType?.startsWith('audio/') ?? false);
  bool get isDocument => type == 'document' || !isImage && !isVideo && !isAudio;

  String get formattedSize {
    if (size == null) return '';
    if (size! < 1024) return '${size}B';
    if (size! < 1024 * 1024) return '${(size! / 1024).toStringAsFixed(1)}KB';
    return '${(size! / (1024 * 1024)).toStringAsFixed(1)}MB';
  }
}

/// A versatile widget for displaying message attachments
///
/// Features:
/// - Support for images, videos, audio, and documents
/// - Thumbnail generation and caching
/// - Progress indicators for uploads/downloads
/// - Customizable appearance
/// - Download and view actions
/// - Error handling and retry
/// - Accessibility support
class MessageAttachmentWidget extends StatefulWidget {
  /// The attachment to display
  final MessageAttachment attachment;

  /// Configuration for appearance
  final AttachmentConfig? config;

  /// Whether the attachment is being uploaded
  final bool isUploading;

  /// Upload/download progress (0.0 to 1.0)
  final double? progress;

  /// Callback when attachment is tapped
  final VoidCallback? onTap;

  /// Callback when download is requested
  final VoidCallback? onDownload;

  /// Callback when remove is requested
  final VoidCallback? onRemove;

  /// Whether to show action buttons
  final bool showActions;

  /// Custom error message
  final String? errorMessage;

  const MessageAttachmentWidget({
    super.key,
    required this.attachment,
    this.config,
    this.isUploading = false,
    this.progress,
    this.onTap,
    this.onDownload,
    this.onRemove,
    this.showActions = true,
    this.errorMessage,
  });

  @override
  State<MessageAttachmentWidget> createState() =>
      _MessageAttachmentWidgetState();
}

class _MessageAttachmentWidgetState extends State<MessageAttachmentWidget>
    with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;
  bool _hasError = false;

  @override
  void initState() {
    super.initState();

    _animationController = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );

    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeOut),
    );

    _animationController.forward();
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final config = widget.config ?? _getDefaultConfig(theme);

    return AnimatedBuilder(
      animation: _fadeAnimation,
      builder: (context, child) {
        return Opacity(
          opacity: _fadeAnimation.value,
          child: Transform.translate(
            offset: Offset(0, 20 * (1 - _fadeAnimation.value)),
            child: _buildAttachmentContainer(context, config),
          ),
        );
      },
    );
  }

  AttachmentConfig _getDefaultConfig(ThemeData theme) {
    return AttachmentConfig(
      backgroundColor: theme.brightness == Brightness.dark
          ? const Color(0xFF374151).withValues(alpha: 0.8)
          : Colors.white.withValues(alpha: 0.9),
      borderColor: theme.brightness == Brightness.dark
          ? const Color(0xFF4B5563).withValues(alpha: 0.5)
          : const Color(0xFFE5E7EB).withValues(alpha: 0.8),
      shadows: [
        BoxShadow(
          color: Colors.black.withValues(alpha: 0.1),
          blurRadius: 8,
          offset: const Offset(0, 2),
        ),
      ],
    );
  }

  Widget _buildAttachmentContainer(
    BuildContext context,
    AttachmentConfig config,
  ) {
    return Container(
      constraints: BoxConstraints(
        maxWidth: config.maxWidth,
        maxHeight: widget.attachment.isImage
            ? config.maxHeight
            : double.infinity,
      ),
      margin: config.margin,
      decoration: BoxDecoration(
        color: config.backgroundColor,
        borderRadius: config.borderRadius,
        border: config.borderColor != null
            ? Border.all(color: config.borderColor!)
            : null,
        boxShadow: config.shadows,
      ),
      child: ClipRRect(
        borderRadius: config.borderRadius,
        child: widget.attachment.isImage
            ? _buildImageAttachment(config)
            : _buildFileAttachment(context, config),
      ),
    );
  }

  Widget _buildImageAttachment(AttachmentConfig config) {
    return Stack(
      children: [
        // Image content
        GestureDetector(
          onTap: widget.onTap,
          child: Container(
            width: double.infinity,
            height: config.maxHeight,
            color: Colors.grey[200],
            child:
                widget.attachment.url != null ||
                    widget.attachment.localPath != null
                ? Image.network(
                    widget.attachment.url ?? widget.attachment.localPath!,
                    fit: BoxFit.cover,
                    errorBuilder: (context, error, stackTrace) {
                      WidgetsBinding.instance.addPostFrameCallback((_) {
                        setState(() {
                          _hasError = true;
                        });
                      });
                      return _buildErrorState();
                    },
                    loadingBuilder: (context, child, loadingProgress) {
                      if (loadingProgress == null) return child;
                      return _buildImageLoadingState(loadingProgress);
                    },
                  )
                : _buildImagePlaceholder(),
          ),
        ),

        // Progress overlay
        if (widget.isUploading || widget.progress != null)
          _buildProgressOverlay(),

        // Actions overlay
        if (widget.showActions && !widget.isUploading) _buildActionsOverlay(),

        // Error overlay
        if (_hasError || widget.errorMessage != null) _buildErrorOverlay(),
      ],
    );
  }

  Widget _buildFileAttachment(BuildContext context, AttachmentConfig config) {
    final theme = Theme.of(context);

    return GestureDetector(
      onTap: widget.onTap,
      child: Container(
        padding: config.padding,
        child: Row(
          children: [
            // File icon
            Container(
              width: 48,
              height: 48,
              decoration: BoxDecoration(
                color: _getFileTypeColor().withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Icon(
                _getFileTypeIcon(),
                color: _getFileTypeColor(),
                size: 24,
              ),
            ),

            const SizedBox(width: 12),

            // File info
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    widget.attachment.name,
                    style: theme.textTheme.bodyMedium?.copyWith(
                      fontWeight: FontWeight.w500,
                    ),
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),
                  const SizedBox(height: 4),
                  Row(
                    children: [
                      if (widget.attachment.formattedSize.isNotEmpty) ...[
                        Text(
                          widget.attachment.formattedSize,
                          style: theme.textTheme.bodySmall?.copyWith(
                            color: theme.textTheme.bodySmall?.color?.withValues(
                              alpha: 0.6,
                            ),
                          ),
                        ),
                        const SizedBox(width: 8),
                      ],
                      if (widget.attachment.type.isNotEmpty)
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 6,
                            vertical: 2,
                          ),
                          decoration: BoxDecoration(
                            color: _getFileTypeColor().withValues(alpha: 0.1),
                            borderRadius: BorderRadius.circular(4),
                          ),
                          child: Text(
                            widget.attachment.type.toUpperCase(),
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: _getFileTypeColor(),
                              fontSize: 10,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                    ],
                  ),

                  // Progress bar for file uploads
                  if (widget.isUploading || widget.progress != null)
                    _buildProgressBar(),
                ],
              ),
            ),

            // Action buttons
            if (widget.showActions && !widget.isUploading) _buildFileActions(),
          ],
        ),
      ),
    );
  }

  Widget _buildImagePlaceholder() {
    return Container(
      width: double.infinity,
      height: double.infinity,
      color: Colors.grey[200],
      child: const Icon(Icons.image, size: 48, color: Colors.grey),
    );
  }

  Widget _buildImageLoadingState(ImageChunkEvent loadingProgress) {
    final progress = loadingProgress.expectedTotalBytes != null
        ? loadingProgress.cumulativeBytesLoaded /
              loadingProgress.expectedTotalBytes!
        : null;

    return Container(
      width: double.infinity,
      height: double.infinity,
      color: Colors.grey[200],
      child: Center(
        child: CircularProgressIndicator(
          value: progress,
          strokeWidth: 3,
          valueColor: const AlwaysStoppedAnimation<Color>(AppTheme.primaryBlue),
        ),
      ),
    );
  }

  Widget _buildProgressOverlay() {
    return Positioned.fill(
      child: Container(
        color: Colors.black.withValues(alpha: 0.5),
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              CircularProgressIndicator(
                value: widget.progress,
                strokeWidth: 3,
                valueColor: const AlwaysStoppedAnimation<Color>(Colors.white),
              ),
              if (widget.progress != null) ...[
                const SizedBox(height: 8),
                Text(
                  '${(widget.progress! * 100).toInt()}%',
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildActionsOverlay() {
    return Positioned(
      top: 8,
      right: 8,
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (widget.onDownload != null)
            _buildActionButton(
              icon: Icons.download,
              onPressed: widget.onDownload!,
              tooltip: 'Download',
            ),
          if (widget.onRemove != null) ...[
            const SizedBox(width: 4),
            _buildActionButton(
              icon: Icons.close,
              onPressed: widget.onRemove!,
              tooltip: 'Remove',
              backgroundColor: Colors.red,
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildErrorOverlay() {
    return Positioned.fill(
      child: Container(
        color: Colors.black.withValues(alpha: 0.7),
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.error_outline, color: Colors.white, size: 32),
              const SizedBox(height: 8),
              Text(
                widget.errorMessage ?? 'Failed to load image',
                style: const TextStyle(color: Colors.white, fontSize: 14),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 12),
              ElevatedButton(
                onPressed: () {
                  setState(() {
                    _hasError = false;
                  });
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.primaryBlue,
                  foregroundColor: Colors.white,
                ),
                child: const Text('Retry'),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildErrorState() {
    return Container(
      width: double.infinity,
      height: double.infinity,
      color: Colors.grey[200],
      child: const Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.broken_image, size: 48, color: Colors.grey),
          SizedBox(height: 8),
          Text(
            'Failed to load image',
            style: TextStyle(color: Colors.grey, fontSize: 12),
          ),
        ],
      ),
    );
  }

  Widget _buildProgressBar() {
    if (widget.progress == null && !widget.isUploading) {
      return const SizedBox.shrink();
    }

    return Container(
      margin: const EdgeInsets.only(top: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          LinearProgressIndicator(
            value: widget.progress,
            backgroundColor: Colors.grey[300],
            valueColor: const AlwaysStoppedAnimation<Color>(
              AppTheme.primaryBlue,
            ),
          ),
          if (widget.progress != null) ...[
            const SizedBox(height: 4),
            Text(
              'Uploading ${(widget.progress! * 100).toInt()}%',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: AppTheme.primaryBlue,
                fontSize: 10,
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildFileActions() {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        if (widget.onDownload != null)
          _buildActionButton(
            icon: Icons.download,
            onPressed: widget.onDownload!,
            tooltip: 'Download',
            compact: true,
          ),
        if (widget.onRemove != null) ...[
          const SizedBox(height: 4),
          _buildActionButton(
            icon: Icons.close,
            onPressed: widget.onRemove!,
            tooltip: 'Remove',
            backgroundColor: Colors.red,
            compact: true,
          ),
        ],
      ],
    );
  }

  Widget _buildActionButton({
    required IconData icon,
    required VoidCallback onPressed,
    required String tooltip,
    Color? backgroundColor,
    bool compact = false,
  }) {
    return Tooltip(
      message: tooltip,
      child: Container(
        decoration: BoxDecoration(
          color:
              backgroundColor?.withValues(alpha: 0.9) ??
              Colors.black.withValues(alpha: 0.7),
          borderRadius: BorderRadius.circular(compact ? 16 : 20),
        ),
        child: Material(
          color: Colors.transparent,
          child: InkWell(
            borderRadius: BorderRadius.circular(compact ? 16 : 20),
            onTap: onPressed,
            child: Padding(
              padding: EdgeInsets.all(compact ? 6 : 8),
              child: Icon(icon, color: Colors.white, size: compact ? 14 : 16),
            ),
          ),
        ),
      ),
    );
  }

  IconData _getFileTypeIcon() {
    if (widget.attachment.isAudio) return Icons.audiotrack;
    if (widget.attachment.isVideo) return Icons.videocam;
    if (widget.attachment.isDocument) {
      if (widget.attachment.mimeType?.contains('pdf') ?? false)
        return Icons.picture_as_pdf;
      if (widget.attachment.mimeType?.contains('word') ?? false)
        return Icons.description;
      if (widget.attachment.mimeType?.contains('excel') ?? false)
        return Icons.table_chart;
      if (widget.attachment.mimeType?.contains('powerpoint') ?? false)
        return Icons.slideshow;
    }
    return Icons.insert_drive_file;
  }

  Color _getFileTypeColor() {
    if (widget.attachment.isAudio) return Colors.orange;
    if (widget.attachment.isVideo) return Colors.red;
    if (widget.attachment.isDocument) {
      if (widget.attachment.mimeType?.contains('pdf') ?? false)
        return Colors.red;
      if (widget.attachment.mimeType?.contains('word') ?? false)
        return Colors.blue;
      if (widget.attachment.mimeType?.contains('excel') ?? false)
        return Colors.green;
      if (widget.attachment.mimeType?.contains('powerpoint') ?? false)
        return Colors.orange;
    }
    return AppTheme.primaryBlue;
  }
}

/// A widget for displaying multiple attachments in a grid or list
class AttachmentGrid extends StatelessWidget {
  final List<MessageAttachment> attachments;
  final AttachmentConfig? config;
  final Function(MessageAttachment)? onAttachmentTap;
  final Function(MessageAttachment)? onDownload;
  final Function(MessageAttachment)? onRemove;
  final bool showActions;
  final int maxColumns;

  const AttachmentGrid({
    super.key,
    required this.attachments,
    this.config,
    this.onAttachmentTap,
    this.onDownload,
    this.onRemove,
    this.showActions = true,
    this.maxColumns = 2,
  });

  @override
  Widget build(BuildContext context) {
    if (attachments.isEmpty) {
      return const SizedBox.shrink();
    }

    // Single attachment
    if (attachments.length == 1) {
      return MessageAttachmentWidget(
        attachment: attachments.first,
        config: config,
        onTap: onAttachmentTap != null
            ? () => onAttachmentTap!(attachments.first)
            : null,
        onDownload: onDownload != null
            ? () => onDownload!(attachments.first)
            : null,
        onRemove: onRemove != null ? () => onRemove!(attachments.first) : null,
        showActions: showActions,
      );
    }

    // Multiple attachments in grid
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: attachments.map((attachment) {
        return SizedBox(
          width: (MediaQuery.of(context).size.width * 0.78) / maxColumns - 4,
          child: MessageAttachmentWidget(
            attachment: attachment,
            config: config?.copyWith(
              maxWidth:
                  (MediaQuery.of(context).size.width * 0.78) / maxColumns - 4,
            ),
            onTap: onAttachmentTap != null
                ? () => onAttachmentTap!(attachment)
                : null,
            onDownload: onDownload != null
                ? () => onDownload!(attachment)
                : null,
            onRemove: onRemove != null ? () => onRemove!(attachment) : null,
            showActions: showActions,
          ),
        );
      }).toList(),
    );
  }
}
