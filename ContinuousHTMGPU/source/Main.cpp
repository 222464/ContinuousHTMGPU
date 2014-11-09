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

	std::vector<bool> actionMask(64 * 64, false);

	actionMask[0] = true;

	htmrl.createRandom(cs, program, 64, 64, layerDescs, actionMask, -0.1f, 0.1f, generator);

	system("pause");

	return 0;
}