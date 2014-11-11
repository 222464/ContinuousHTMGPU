#pragma once

#include <SFML/Graphics.hpp>
#include <htm/HTMRL.h>

namespace vis {
	class HTMRLVisualizer {
	private:
		sf::RenderTexture _rt;
	public:
		void create(unsigned int width);

		void update(sf::RenderTexture &target, const sf::Vector2f &position, const sf::Vector2f &scale, sys::ComputeSystem &cs, const htm::HTMRL &htmrl, std::mt19937 &generator);
	};
}