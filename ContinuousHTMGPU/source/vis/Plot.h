#pragma once

#include <SFML/Graphics.hpp>

namespace vis {
	struct Point {
		sf::Vector2f _position;

		sf::Color _color;

		Point()
			: _color(sf::Color::Black)
		{}
	};

	struct Curve {
		std::string _name;

		std::vector<Point> _points;
	};

	struct Plot {
		sf::Color _axesColor;
		sf::Color _backgroundColor;

		std::vector<Curve> _curves;

		Plot()
			: _axesColor(sf::Color::Black), _backgroundColor(sf::Color::White)
		{}

		void draw(sf::RenderTarget &target, const sf::Texture &lineGradientTexture, const sf::Font &tickFont, float tickTextScale,
			const sf::Vector2f &domain, const sf::Vector2f &range, const sf::Vector2f &margins, const sf::Vector2f &tickIncrements, float axesSize, float lineSize, float tickSize, float tickLength, float textTickOffset, int precision);
	};

	float vectorMagnitude(const sf::Vector2f &vector);
	sf::Vector2f vectorNormalize(const sf::Vector2f &vector);
}