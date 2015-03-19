using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;

namespace ConvolutionalNeuralNetworks
{
	namespace Activations
	{
		interface IActivation
		{
			/// <summary>
			/// f(x)
			/// </summary>
			/// <param name="x"></param>
			/// <returns></returns>
			double f(double x);
			/// <summary>
			/// f'(x)
			/// </summary>
			/// <param name="x"></param>
			/// <returns></returns>
			double df(double x);
			/// <summary>
			/// 活性化関数のType
			/// </summary>
			/// <returns></returns>
			string Type();
		}
		/// <summary>
		/// f(x) = x
		/// </summary>
		public class Identity : IActivation
		{
			public double f(double x) { return x; }
			public double df(double x) { return 1; }
			public string Type() { return "Identity"; }
		}
		/// <summary>
		/// f(x) = 1/(1 + e^{-x})
		/// </summary>
		public class Sigmoid : IActivation
		{
			public double f(double x) { return 1.0 / (1.0 + Math.Exp(-x)); }
			public double df(double x)
			{
				double f_x = f(x);
				return f_x * (1.0 - f_x);
			}
			public string Type() { return "Sigmoid"; }
		}
		/// <summary>
		/// f(x) = tanh(x)
		/// </summary>
		public class Tanh : IActivation
		{
			public double f(double x) { return Math.Tanh(x); }
			public double df(double x)
			{
				double f_x = f(x);
				return 1.0 - f_x * f_x;
			}
			public string Type() { return "Tanh"; }
		}
		/// <summary>
		/// f(x) = max(x,0)
		/// </summary>
		public class ReLU : IActivation
		{
			public double f(double x) { return Math.Max(0, x); }
			public double df(double x) { return x > 0 ? 1.0 : 0.0; }
			public string Type() { return "ReLU"; }
		}
	}
}