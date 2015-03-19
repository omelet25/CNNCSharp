using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;

namespace ConvolutionalNeuralNetworks
{
	namespace ElementWises
	{
		interface IElementWise
		{
			/// <summary>
			/// f(x)
			/// </summary>
			/// <param name="ms"></param>
			/// <param name="y"></param>
			/// <param name="x"></param>
			/// <returns></returns>
			double f(Matrix<double>[] ms, int y, int x);
			/// <summary>
			/// f'(x)
			/// </summary>
			/// <param name="ms"></param>
			/// <returns></returns>
			double[] df(double[] ms);
			/// <summary>
			/// element wise type
			/// </summary>
			/// <returns></returns>
			string Type();
		}
		/// <summary>
		/// 各入力の最大値を出力
		/// </summary>
		public class MaxOut : IElementWise
		{
			public double f(Matrix<double>[] ms, int y, int x) { return ms.Max(_ => _[y, x]); }
			public double[] df(double[] ms)
			{
				double _max = ms[0];
				int _idx = 0;
				for (int i = 1; i < ms.Length; i++)
				{
					if (_max < ms[i]) { _idx = i; _max = ms[i]; }
				}
				double[] _df = new double[ms.Length];
				_df[_idx] = 1;
				return (double[])_df.Clone();
			}
			public string Type() { return "MaxOut"; }
		}
		/// <summary>
		/// 各入力の加重総和を出力
		/// </summary>
		public class Average : IElementWise
		{
			/// <summary>
			/// Weight Sum
			/// </summary>
			/// <param name="coeffs">
			/// <para>各要素に掛かる係数</para>
			/// <para>nullの場合は平均(1.0 / ms.Length)</para>
			/// </param>
			/// <returns></returns>
			public double f(Matrix<double>[] ms, int y, int x) { return ms.Average(_ => _[y, x]); }
			public double[] df(double[] ms)
			{
				double[] _df = new double[ms.Length];
				for (int i = 0; i < _df.Length; i++) { _df[i] = 1.0 / _df.Length; }
				return (double[])_df.Clone();
			}
			public string Type() { return "Average"; }
		}
	}

}