using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;

namespace ConvolutionalNeuralNetworks
{

	namespace Poolings
	{
		interface IPooling
		{
			/// <summary>
			/// f(x)
			/// </summary>
			/// <param name="m"></param>
			/// <returns></returns>
			double f(Matrix<double> m);
			/// <summary>
			/// f'(x)
			/// </summary>
			/// <param name="m"></param>
			/// <returns></returns>
			Matrix<double> df(Matrix<double> m);
			/// <summary>
			/// pooling types
			/// </summary>
			/// <returns></returns>
			string Type();
		}
		public class Max : IPooling
		{
			public double f(Matrix<double> m) { return m.Enumerate().Max(); }
			public Matrix<double> df(Matrix<double> m)
			{
				double _max = -double.MaxValue;
				int _maxi = 0, _maxj = 0;
				for (int i = 0; i < m.RowCount; i++)
				{
					for (int j = 0; j < m.ColumnCount; j++)
					{
						if (_max < m[i, j]) { _max = m[i, j]; _maxi = i; _maxj = j; }
					}
				}
				
				return Matrix<double>.Build.Dense(
					m.RowCount, m.ColumnCount, 
					new Func<int, int, double>((i, j) => { return i == _maxi && j == _maxj ? 1 : 0; })
				);
			}
			public string Type() { return "Max"; }
		}
		public class Average : IPooling
		{
			public double f(Matrix<double> m) { return m.Enumerate().Average(); }
			public Matrix<double> df(Matrix<double> m)
			{
				double N = m.RowCount * m.ColumnCount;
				return m / N;
			}
			public string Type() { return "Average"; }
		}
	}

}