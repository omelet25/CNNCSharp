using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;
using MathNet.Numerics.LinearAlgebra;

using ConvolutionalNeuralNetworks.Tools;
using ConvolutionalNeuralNetworks.Layers;

namespace ConvolutionalNeuralNetworks
{
	class Program
	{
		static void Main(string[] args)
		{
			//Test.MNIST mnist = new Test.MNIST();
			//mnist.Start(0);

			//foreach (var eta in new double[] { 0.01 })
			//{
			//	Test.HUMAN_EXTRACTION human = new Test.HUMAN_EXTRACTION(96, 48);
			//	human._eta = eta;
			//	human.Start(1);
			//}
			Test.HUMAN_DIRECTION direction = new Test.HUMAN_DIRECTION(96, 48);
			direction.Start();
		}
	}
}
