using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;

namespace ConvolutionalNeuralNetworks
{
	namespace LossFunctions
	{
		interface ILossFunction
		{
			/// <summary>
			/// f(x) : error
			/// </summary>
			/// <param name="y"></param>
			/// <param name="t"></param>
			/// <returns></returns>
			double f(double y, double t);
			/// <summary>
			/// f'(x)
			/// </summary>
			/// <param name="y"></param>
			/// <param name="t"></param>
			/// <returns></returns>
			double df(double y, double t);
			/// <summary>
			/// 誤差関数 type
			/// </summary>
			/// <returns></returns>
			string Type();
		}

		/// <summary>
		/// <para>平均二乗誤差</para>
		/// </summary>
		public class MSE : ILossFunction
		{
			public double f(double y, double t)
			{
				return (y - t) * (y - t) / 2.0;
			}
			public double df(double y, double t)
			{
				return y - t;
			}
			public string Type() { return "MSE"; }
		}

		/// <summary>
		/// <para>誤差関数</para>
		/// <para>多クラス交差エントロピー</para>
		/// </summary>
		public class MultiCrossEntropy : ILossFunction
		{
			public double f(double y, double t)
			{
				if (y == t || t == 0) { return 0; }
				return -(t * Math.Log(y));
			}
			public double df(double y, double t)
			{
				return y - t;
			}
			public string Type() { return "MultiClassCrossEntropy"; }
		}

		public class BinaryCrossEntropy : ILossFunction
		{
			public double f(double y, double t)
			{
				if (y == t) { return 0; }
				else
				{;
				return -(t * Math.Log(y) + (1.0 - t) * Math.Log(1.0 - y));
				}
			}
			public double df(double y, double t)
			{
				return (y - t);
			}
			public string Type() { return "BinaryCrossEntropy"; }
		}
	}
}