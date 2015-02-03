#include "HTMRLVisualizer.h"

using namespace vis;

void HTMRLVisualizer::create(unsigned int width) {
	_rt.create(width, width, false);
	_rt.setSmooth(true);
}

void HTMRLVisualizer::update(sf::RenderTexture &target, const sf::Vector2f &position, const sf::Vector2f &scale, sys::ComputeSystem &cs, const htm::HTMRL &htmrl, std::mt19937 &generator) {
	std::vector<std::shared_ptr<sf::Image>> images;

	htmrl.exportCellData(cs, images, 5634);

	const float heightStep = 1.0f;
	const float transparency = 0.3f;
	const int cellLayerSteps = 5;

	int h = 0;

	sf::Texture imageTexture;

	for (int i = 0; i < images.size(); i++) {
		// Render to RT
		_rt.setActive();

		imageTexture.loadFromImage(*images[i]);

		imageTexture.setSmooth(false);
		
		sf::Sprite imageSprite;
		imageSprite.setTexture(imageTexture);

		imageSprite.setOrigin(imageTexture.getSize().x * 0.5f, imageTexture.getSize().y * 0.5f);

		imageSprite.setRotation(45.0f);
		imageSprite.setPosition(_rt.getSize().x * 0.5f, _rt.getSize().y * 0.5f);
		imageSprite.setScale(static_cast<float>(_rt.getSize().x) / imageTexture.getSize().x * 0.75f, static_cast<float>(_rt.getSize().y) / imageTexture.getSize().y * 0.75f);

		sf::RenderStates clearStates;
		clearStates.blendMode = sf::BlendNone;

		sf::RectangleShape clearShape;
		clearShape.setSize(sf::Vector2f(_rt.getSize().x, _rt.getSize().y));
		clearShape.setFillColor(sf::Color::Transparent);

		_rt.draw(clearShape, clearStates);

		_rt.draw(imageSprite);

		_rt.display();

		// Render rt to main image
		target.setActive();

		sf::Sprite transformedSprite;
		transformedSprite.setTexture(_rt.getTexture());
		transformedSprite.setOrigin(transformedSprite.getTexture()->getSize().x * 0.5f, transformedSprite.getTexture()->getSize().y * 0.5f);
	
		transformedSprite.setScale(scale.x * 0.5f, scale.y * 0.25f);
		transformedSprite.setColor(sf::Color(255, 255, 255, 255.0f * transparency));

		target.setSmooth(true);

		for (int s = 0; s < cellLayerSteps; s++) {
			transformedSprite.setPosition(position.x, position.y - h * heightStep);
			target.draw(transformedSprite);

			h++;
		}
	}

	target.display();
}