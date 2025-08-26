import 'package:flutter/material.dart';

import '../models/message.dart';

/// Welcome prompts data - same as

class WelcomeSection extends StatelessWidget {
  const WelcomeSection({super.key, required this.sessionId});

  final String sessionId;

  List<WelcomePrompt> get welcomePrompts => [
    WelcomePrompt(icon: '🏋️', text: 'Create a personalized workout plan', category: 'Workout'),
    WelcomePrompt(icon: '🥗', text: 'Get nutrition and meal planning advice', category: 'Nutrition'),
    WelcomePrompt(icon: '📊', text: 'Track my fitness progress', category: 'Progress'),
    WelcomePrompt(icon: '💡', text: 'Learn proper exercise techniques', category: 'Education'),
    WelcomePrompt(icon: '🎯', text: 'Set and achieve fitness goals', category: 'Goals'),
    WelcomePrompt(icon: '🧘', text: 'Recovery and wellness tips', category: 'Wellness'),
  ];

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 24),
      child: Column(
        children: [
          Text(
            'How can I help you reach your fitness goals?',
            style: Theme.of(context).textTheme.headlineMedium?.copyWith(fontSize: 20, fontWeight: FontWeight.w600),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 16),
          GridView.builder(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: _getCrossAxisCount(context),
              childAspectRatio: 1.2,
              crossAxisSpacing: 12,
              mainAxisSpacing: 12,
            ),
            itemCount: welcomePrompts.length,
            itemBuilder: (context, index) {
              final prompt = welcomePrompts[index];

              return _WelcomePromptCard(
                prompt: prompt,
                onTap: () {
                  // sendWelcomePrompt
                  // provider.sendWelcomePrompt(sessionId, prompt)
                },
              );
            },
          ),
        ],
      ),
    );
  }

  int _getCrossAxisCount(BuildContext context) {
    final width = MediaQuery.of(context).size.width;
    if (width > 1200) return 3;
    if (width > 600) return 2;
    return 1;
  }
}

class _WelcomePromptCard extends StatefulWidget {
  final WelcomePrompt prompt;
  final VoidCallback? onTap;

  const _WelcomePromptCard({required this.prompt, this.onTap});

  @override
  State<_WelcomePromptCard> createState() => _WelcomePromptCardState();
}

class _WelcomePromptCardState extends State<_WelcomePromptCard> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _scaleAnimation;
  bool _isHovered = false;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(duration: const Duration(milliseconds: 200), vsync: this);
    _scaleAnimation = Tween<double>(
      begin: 1.0,
      end: 1.05,
    ).animate(CurvedAnimation(parent: _controller, curve: Curves.easeInOut));
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      onEnter: (_) {
        setState(() => _isHovered = true);
        _controller.forward();
      },
      onExit: (_) {
        setState(() => _isHovered = false);
        _controller.reverse();
      },
      child: AnimatedBuilder(
        animation: _scaleAnimation,
        builder: (context, child) {
          return Transform.scale(
            scale: _scaleAnimation.value,
            child: Material(
              color: Colors.transparent,
              borderRadius: BorderRadius.circular(16),
              child: InkWell(
                onTap: widget.onTap,
                borderRadius: BorderRadius.circular(16),
                child: Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: _isHovered
                        ? (Theme.of(context).brightness == Brightness.dark
                              ? const Color(0xFF1F2937) // hover:bg-gray-800
                              : const Color(0xFFDEEEFF)) // hover:bg-blue-50
                        : (Theme.of(context).brightness == Brightness.dark
                              ? const Color(0xFF1F2937).withValues(alpha: .5) // bg-gray-800/50
                              : Colors.white),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(
                      color: _isHovered
                          ? const Color(0xFF3B82F6) // hover:border-fitvise-primary
                          : (Theme.of(context).brightness == Brightness.dark
                                ? const Color(0xFF4B5563) // border-gray-600
                                : const Color(0xFFD1D5DB)), // border-gray-300
                      style: BorderStyle.solid,
                      width: 2,
                    ),
                  ),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(widget.prompt.icon, style: const TextStyle(fontSize: 32)),
                      const SizedBox(height: 8),
                      Text(
                        widget.prompt.text,
                        style: Theme.of(context).textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w500),
                        textAlign: TextAlign.left,
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                      const SizedBox(height: 4),
                      Align(
                        alignment: Alignment.centerLeft,
                        child: Text(
                          widget.prompt.category,
                          style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: Theme.of(context).brightness == Brightness.dark
                                ? const Color(0xFF9CA3AF) // text-gray-400
                                : const Color(0xFF6B7280), // text-gray-500
                          ),
                          textAlign: TextAlign.left,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}
