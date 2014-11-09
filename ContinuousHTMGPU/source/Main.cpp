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

	float reward = 0.0f;
	float prevReward = 0.0f;

	float initReward = 0.0f;

	float totalReward = 0.0f;

	sf::RenderWindow window;

	window.create(sf::VideoMode(800, 600), "Pole Balancing");

	window.setVerticalSyncEnabled(true);

	//window.setFramerateLimit(60);

	// -------------------------- Load Resources --------------------------

	sf::Texture backgroundTexture;
	sf::Texture cartTexture;
	sf::Texture poleTexture;

	backgroundTexture.loadFromFile("resources/background.png");
	cartTexture.loadFromFile("resources/cart.png");
	poleTexture.loadFromFile("resources/pole.png");

	sf::Texture inputCartTexture;
	sf::Texture inputPoleTexture;

	inputCartTexture.loadFromFile("resources/inputCart.png");
	inputPoleTexture.loadFromFile("resources/inputPole.png");

	// --------------------------------------------------------------------

	sf::Sprite backgroundSprite;
	sf::Sprite cartSprite;
	sf::Sprite poleSprite;

	backgroundSprite.setTexture(backgroundTexture);
	cartSprite.setTexture(cartTexture);
	poleSprite.setTexture(poleTexture);

	backgroundSprite.setPosition(sf::Vector2f(0.0f, 0.0f));

	cartSprite.setOrigin(sf::Vector2f(static_cast<float>(cartSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(cartSprite.getTexture()->getSize().y)));
	poleSprite.setOrigin(sf::Vector2f(static_cast<float>(poleSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(poleSprite.getTexture()->getSize().y)));

	sf::Sprite inputCartSprite;
	sf::Sprite inputPoleSprite;

	inputCartSprite.setTexture(inputCartTexture);
	inputPoleSprite.setTexture(inputPoleTexture);

	inputCartSprite.setOrigin(sf::Vector2f(static_cast<float>(inputCartSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(inputCartSprite.getTexture()->getSize().y)));
	inputPoleSprite.setOrigin(sf::Vector2f(static_cast<float>(inputPoleSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(inputPoleSprite.getTexture()->getSize().y)));

	// ----------------------------- Physics ------------------------------

	float pixelsPerMeter = 128.0f;
	float inputPixelsPerMeter = 8.0f;
	float poleLength = 1.0f;
	float g = -2.8f;
	float massMass = 40.0f;
	float cartMass = 2.0f;
	sf::Vector2f massPos(0.0f, poleLength);
	sf::Vector2f massVel(0.0f, 0.0f);
	float poleAngle = static_cast<float>(3.14159f) * 0.0f;
	float poleAngleVel = 0.0f;
	float poleAngleAccel = 0.0f;
	float cartX = 0.0f;
	float cartVelX = 0.0f;
	float cartAccelX = 0.0f;
	float poleRotationalFriction = 0.008f;
	float cartMoveRadius = 1.8f;
	float cartFriction = 0.02f;
	float maxSpeed = 3.0f;

	// ---------------------------- Game Loop -----------------------------

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	float fitness = 0.0f;
	float prevFitness = 0.0f;

	float lowPassFitness = 0.0f;

	bool reverseDirection = false;

	bool trainMode = true;

	bool tDownLastFrame = false;

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	sf::Font font;

	font.loadFromFile("Resources/pixelated.ttf");

	sf::RenderTexture inputRT;

	inputRT.create(64, 32);

	float plotMin = 99999.0f;
	float plotMax = -99999.0f;

	float avgReward = 0.0f;
	float avgRewardDecay = 0.003f;

	float totalTime = 0.0f;

	float plotUpdateTimer = 0.0f;

	htm::HTMRL agent;

	std::vector<htm::HTMRL::LayerDesc> layerDescs(1);

	layerDescs[0]._width = 32;
	layerDescs[0]._height = 32;

	std::vector<bool> actionMask(6, false);

	actionMask[4] = actionMask[5] = true;

	agent.createRandom(cs, program, 2, 3, layerDescs, actionMask, -0.1f, 0.1f, generator);

	std::vector<float> prevInput(6, 0.0f);

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent))
		{
			switch (windowEvent.type)
			{
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		// Update fitness
		if (poleAngle < static_cast<float>(3.14159f))
			fitness = -(static_cast<float>(3.14159f)* 0.5f - poleAngle);
		else
			fitness = -(static_cast<float>(3.14159f)* 0.5f - (static_cast<float>(3.14159f)* 2.0f - poleAngle));

		fitness += static_cast<float>(3.14159f)* 0.5f;

		//fitness = fitness - std::abs(poleAngleVel * 1.0f);

		//fitness = -std::abs(cartX);

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
			fitness = -cartX;
		else if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
			fitness = cartX;

		// ------------------------------ AI -------------------------------

		float dFitness = fitness - prevFitness;

		//reward = dFitness * 5.0f;

		reward = fitness * 0.2f;

		if (totalTime == 0.0f)
			avgReward = reward;
		else
			avgReward = (1.0f - avgRewardDecay) * avgReward + avgRewardDecay * reward;

		sf::Image img = inputRT.getTexture().copyToImage();

		std::vector<float> state(6);

		agent.setInput(0, cartX * 0.25f);
		agent.setInput(1, cartVelX * 0.4f);
		agent.setInput(2, std::fmod(poleAngle + static_cast<float>(3.14159f), 2.0f * static_cast<float>(3.14159f)) / (2.0f * static_cast<float>(3.14159f)) * 2.0f - 1.0f);
		agent.setInput(3, poleAngleVel * 0.2f);
		agent.setInput(4, prevInput[4]);
		agent.setInput(5, prevInput[5]);

		agent.step(cs, reward, 0.01f, 0.01f, 0.05f, 4, 0.05f, 0.8f, 0.5f, 0.99f, 0.05f, 0.05f, generator);

		prevInput[4] = agent.getOutput(4);
		prevInput[5] = agent.getOutput(5);

		float dir = prevInput[4];

		float agentForce = 4000.0f * dir;
	
		prevFitness = fitness;

		// ---------------------------- Physics ----------------------------

		float pendulumCartAccelX = cartAccelX;

		if (cartX < -cartMoveRadius)
			pendulumCartAccelX = 0.0f;
		else if (cartX > cartMoveRadius)
			pendulumCartAccelX = 0.0f;

		poleAngleAccel = pendulumCartAccelX * std::cos(poleAngle) + g * std::sin(poleAngle);
		poleAngleVel += -poleRotationalFriction * poleAngleVel + poleAngleAccel * dt;
		poleAngle += poleAngleVel * dt;

		massPos = sf::Vector2f(cartX + std::cos(poleAngle + static_cast<float>(3.14159f)* 0.5f) * poleLength, std::sin(poleAngle + static_cast<float>(3.14159f)* 0.5f) * poleLength);

		float force = 0.0f;

		if (std::abs(cartVelX) < maxSpeed) {
			force = std::max<float>(-4000.0f, std::min<float>(4000.0f, agentForce));

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
				force = -4000.0f;

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
				force = 4000.0f;
		}

		//if (trainMode) {
		//	en.calculateGradient(std::vector<float>(1, force / 4000.0f));

		//	en.moveAlongGradient(0.01f);
		//}

		//en.updateContext();

		if (cartX < -cartMoveRadius) {
			cartX = -cartMoveRadius;

			cartAccelX = -cartVelX / dt;
			cartVelX = -0.5f * cartVelX;
		}
		else if (cartX > cartMoveRadius) {
			cartX = cartMoveRadius;

			cartAccelX = -cartVelX / dt;
			cartVelX = -0.5f * cartVelX;
		}

		cartAccelX = 0.25f * (force + massMass * poleLength * poleAngleAccel * std::cos(poleAngle) - massMass * poleLength * poleAngleVel * poleAngleVel * std::sin(poleAngle)) / (massMass + cartMass);
		cartVelX += -cartFriction * cartVelX + cartAccelX * dt;
		cartX += cartVelX * dt;

		poleAngle = std::fmod(poleAngle, (2.0f * static_cast<float>(3.14159f)));

		if (poleAngle < 0.0f)
			poleAngle += static_cast<float>(3.14159f)* 2.0f;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::T)) {
			if (!tDownLastFrame) {
				trainMode = !trainMode;
			}

			tDownLastFrame = true;
		}
		else
			tDownLastFrame = false;

		// ---------------------------- Rendering ----------------------------

		// Render to input buffer
		inputRT.clear();

		inputCartSprite.setPosition(sf::Vector2f(inputRT.getSize().x * 0.5f + inputPixelsPerMeter * cartX, inputRT.getSize().y * 0.5f + 4.0f));

		inputRT.draw(inputCartSprite);

		inputPoleSprite.setPosition(inputCartSprite.getPosition() + sf::Vector2f(0.0f, -4.0f));
		inputPoleSprite.setRotation(poleAngle * 180.0f / static_cast<float>(3.14159f) + 180.0f);

		inputRT.draw(inputPoleSprite);

		inputRT.display();

		window.clear();

		window.draw(backgroundSprite);

		cartSprite.setPosition(sf::Vector2f(800.0f * 0.5f + pixelsPerMeter * cartX, 600.0f * 0.5f + 3.0f));

		window.draw(cartSprite);

		poleSprite.setPosition(cartSprite.getPosition() + sf::Vector2f(0.0f, -45.0f));
		poleSprite.setRotation(poleAngle * 180.0f / static_cast<float>(3.14159f) + 180.0f);

		window.draw(poleSprite);

		sf::Sprite inputSprite;

		inputSprite.setTexture(inputRT.getTexture());

		inputSprite.setPosition(0, 0);
		inputSprite.setScale(4.0f, 4.0f);

		window.draw(inputSprite);

		// -------------------------------------------------------------------

		window.display();

		//dt = clock.getElapsedTime().asSeconds();

		totalTime += dt;
		plotUpdateTimer += dt;
	} while (!quit);

	return 0;
}