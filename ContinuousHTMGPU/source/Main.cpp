#include <cae/ConvAutoEncoder.h>
#include <htm/HTMRL.h>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <time.h>

int main() {
	std::mt19937 generator(time(nullptr));

	sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_gpu);

	sys::ComputeProgram program;

	program.loadFromFile("resources/htmrl.cl", cs);

	htm::HTMRL htmrl;

	std::vector<htm::HTMRL::LayerDesc> layerDescs(1);

	layerDescs[0]._width = 64;
	layerDescs[0]._height = 64;

	//layerDescs[1]._width = 64;
	//layerDescs[1]._height = 64;

	//layerDescs[2]._width = 64;
	//layerDescs[2]._height = 64;

	htmrl.createRandom(cs, program, 64, 64, layerDescs, -0.1f, 0.1f, generator);

	/*htm::HTMFeatureExtractor fExt;

	std::vector<htm::HTMFeatureExtractor::LayerDesc> layerDescs(3);

	layerDescs[0]._width = 128;
	layerDescs[0]._height = 128;
	layerDescs[0]._receptiveFieldRadius = 2;

	layerDescs[1]._width = 64; 
	layerDescs[1]._height = 64;
	layerDescs[1]._receptiveFieldRadius = 2;

	layerDescs[2]._width = 32;
	layerDescs[2]._height = 32;
	layerDescs[2]._receptiveFieldRadius = 2;*/

	float t = 0;
	 
	float dotRadius = 16.0f;

	bool quit = false;

	sf::Clock clock;

	sf::RenderWindow window;

	window.create(sf::VideoMode(512, 512), "HTM", sf::Style::Default);

	//window.setFramerateLimit(60);

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent)) {
			switch (windowEvent.type) {
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::P)) {
			// Save cell data
			htmrl.exportCellData(cs, "data");
		}

		float dotX = 32.0f;// std::cos(t * 4.0f) * 15.0f + 64.0f;
		float dotY = 32.0f;

		for (int x = 0; x < 64; x++)
		for (int y = 0; y < 64; y++) {
			//float r = inImage.getPixel(x, y).r / 255.0f;
			//float g = inImage.getPixel(x, y).g / 255.0f;
			//float b = inImage.getPixel(x, y).b / 255.0f;

			float dist = std::sqrt(static_cast<float>(std::pow(dotX - x, 2) + std::pow(dotY - y, 2)));

			float greyScale = std::max<float>(0.0f, std::min<float>(1.0f, (dotRadius - dist) / dotRadius)) * (std::cos(t) + 0.5f + 0.5f);

			htmrl.setInput(x, y, greyScale);
		}

		if (!sf::Keyboard::isKeyPressed(sf::Keyboard::L)) {
			for (int i = 0; i < 1; i++)
				htmrl.step(cs, 0.05f, 0.05f);

			t += 0.05f;

			if (t > 6.28f)
				t = 0.0f;
		}

		sf::Image outImage;

		outImage.create(htmrl.getLayerDescs().back()._width, htmrl.getLayerDescs().back()._height);

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::O)) {
			for (int x = 0; x < htmrl.getLayerDescs().back()._width; x++)
			for (int y = 0; y < htmrl.getLayerDescs().back()._height; y++) {
				float outputR = htmrl.getPrediction(x, y);

				outImage.setPixel(x, y, sf::Color(outputR * 255.0f, outputR * 255.0f, outputR * 255.0f, 255));
			}
		}
		else {
			for (int x = 0; x < htmrl.getLayerDescs().back()._width; x++)
			for (int y = 0; y < htmrl.getLayerDescs().back()._height; y++) {
				float outputR = htmrl.getSDR(x, y);

				outImage.setPixel(x, y, sf::Color(outputR * 255.0f, outputR * 255.0f, outputR * 255.0f, 255));
			}
		}

		//outImage.saveToFile("out.png");

		window.clear();

		sf::Texture texture;

		texture.loadFromImage(outImage);

		sf::Sprite sprite;

		sprite.setTexture(texture);

		sprite.setScale(512.0f / htmrl.getLayerDescs().back()._width, 512.0f / htmrl.getLayerDescs().back()._height);

		window.draw(sprite);

		window.display();
	} while (!quit);

	return 0;
}