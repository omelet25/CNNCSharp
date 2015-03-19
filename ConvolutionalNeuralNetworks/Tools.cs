using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

using MathNet.Numerics.LinearAlgebra;

namespace ConvolutionalNeuralNetworks
{
	namespace Tools
	{
		static class Utilities
		{
			public static Vector<double> LoadWeightOrBiaseFromFile(string filename)
			{
				using (StreamReader sr = new StreamReader(filename))
				{
					string header = sr.ReadLine();
					if (header != "#Kernels" && header != "#Weights" && header != "#Biases") { throw new FormatException("Read file format error"); }
					int[] _sizes = sr.ReadLine().Split('\t').Select(_ => int.Parse(_)).ToArray();
					int _size = 0;
					if (_sizes.Length == 1) { _size = _sizes[0]; }
					else if (_sizes.Length == 2) { _size = _sizes[0] * _sizes[1]; }
					else { _size = _sizes[0] * _sizes[1] * _sizes[2]; }
					Vector<double> _weight = Vector<double>.Build.Dense(_size);
					int idx = 0;
					while (!sr.EndOfStream)
					{
						double _val;
						string[] _val_str = sr.ReadLine().Split('\t');
						foreach (var _str in _val_str)
						{
							if (double.TryParse(_str, out _val)) { _weight[idx] = _val; idx++; }
						}
					}
					return _weight;
				}
			}

			/// <summary>
			/// [min,max)間の重複のないランダムな整数列を得る
			/// </summary>
			/// <param name="min">min</param>
			/// <param name="max">max</param>
			/// <param name="size">整数列の長さ</param>
			public static int[] RandomIndex(int min, int max, int size)
			{
				HashSet<int> _rand_idx = new HashSet<int>();
				Random _rand = new Random();
				if (size <= 0 || size > max - min) { size = max - min; }
				while (_rand_idx.Count < size) { _rand_idx.Add(_rand.Next(min, max)); }
				return _rand_idx.ToArray();
			}

			public static bool LoadDataList(string dl_fname, out Vector<double>[] input, out Vector<double>[] output,
				bool shuffle = true,string w_infname = null,string w_outfname = null)
			{
				input = null;
				output = null;

				string[] data_type;
				int[] data_length;
				string[] in_list;
				string[] out_list;

				using (System.IO.StreamReader sr_dl = new StreamReader(dl_fname))
				{
					data_type = sr_dl.ReadLine().Split('\t');
					if (data_type.Length != 2) { return false; }

					try { data_length = sr_dl.ReadLine().Split('\t').Select(str => int.Parse(str)).ToArray(); }
					catch (Exception) { return false; }

					if (!(data_type[0] == "data" && data_type[1] == "label" || data_type[0] == "data" && data_type[1] == "data"))
					{
						return false;
					}

					in_list = new string[data_length[0]];
					out_list = new string[data_length[0]];
					input = new Vector<double>[data_length[0]];
					output = new Vector<double>[data_length[0]];

					int[] idx;
					// shuffle する場合はランダムなindex生成
					if (shuffle)
					{
						idx = RandomIndex(0, data_length[0], data_length[0]);
					}
					else
					{
						idx = Enumerable.Range(0, data_length[0]).ToArray();
					}

					for (int i = 0; !sr_dl.EndOfStream; i++)
					{
						var lists = sr_dl.ReadLine().Split('\t');
						in_list[idx[i]] = lists[0];
						out_list[idx[i]] = lists[1];
					}
				}

				// input
				for (int i = 0; i < in_list.Length; i++)
				{
					using (StreamReader sr_data = new StreamReader(in_list[i]))
					{
						input[i] = Vector<double>.Build.DenseOfEnumerable(sr_data.ReadLine().Split('\t').Select(str => double.Parse(str)));
					}
				}

				// output
				if (data_type[1] == "data")
				{
					for (int i = 0; i < out_list.Length; i++)
					{
						using (StreamReader sr_data = new StreamReader(out_list[i]))
						{
							output[i] = Vector<double>.Build.DenseOfEnumerable(sr_data.ReadLine().Split('\t').Select(str => double.Parse(str)));
						}
					}
				}
				else if (data_type[1] == "label")
				{
					for (int i = 0; i < out_list.Length; i++)
					{
						output[i] = Vector<double>.Build.Dense(data_length[1], (idx) => { return int.Parse(out_list[i]) == idx ? 1 : 0; });
					}
				}

				// mean and std datas
				if (w_infname != null)
				{
					Vector<double> m;
					Vector<double> s;
					using (StreamReader sr_w = new StreamReader(w_infname))
					{
						m = Vector<double>.Build.DenseOfEnumerable(sr_w.ReadLine().Split('\t').Select(str => double.Parse(str)));
						s = Vector<double>.Build.DenseOfEnumerable(sr_w.ReadLine().Split('\t').Select(str => double.Parse(str)));
					}
					for (int i = 0; i < input.Length; i++)
					{
						input[i] = (input[i] - m) / s.Map<double>((val) => val == 0 ? 1 : 0);
					}
				}
				if (w_outfname != null && data_type[1] == "data")
				{
					Vector<double> m;
					Vector<double> s;
					using (StreamReader sr_w = new StreamReader(w_outfname))
					{
						m = Vector<double>.Build.DenseOfEnumerable(sr_w.ReadLine().Split('\t').Select(str => double.Parse(str)));
						s = Vector<double>.Build.DenseOfEnumerable(sr_w.ReadLine().Split('\t').Select(str => double.Parse(str)));
					}
					for (int i = 0; i < input.Length; i++)
					{
						output[i] = (output[i] - m) / s.Map<double>((val) => val == 0 ? 1 : 0);
					}
				}

				return true;
			}
		}

		/// <summary>
		/// 型変換クラス
		/// </summary>
		static class Converters
		{
			/// <summary>
			/// Matrix[] → Vector
			/// </summary>
			/// <param name="matrices"></param>
			/// <returns></returns>
			static public Vector<double> ToVector(Matrix<double>[] matrices, int h, int w)
			{
				double[] _vec = new double[matrices.Length * h * w];
				int _idx = 0;
				foreach (var _ in matrices)
				{
					for (int y = 0; y < _.RowCount; y++)
					{
						for (int x = 0; x < _.ColumnCount; x++)
						{
							_vec[_idx] = _[y, x];
							_idx++;
						}
					}
				}

				return Vector<double>.Build.Dense(_vec);
			}

			/// <summary>
			/// Matrix → Vector
			/// </summary>
			/// <param name="matrix"></param>
			/// <returns></returns>
			static public Vector<double> ToVector(Matrix<double> matrix)
			{
				double[] _vec = new double[matrix.RowCount * matrix.ColumnCount];
				int _idx = 0;
				for (int y = 0; y < matrix.RowCount; y++)
				{
					for (int x = 0; x < matrix.ColumnCount; x++)
					{
						_vec[_idx] = matrix[y, x];
						_idx++;
					}
				}
				return Vector<double>.Build.Dense(_vec);
			}

			/// <summary>
			/// Vector → Matrix[]
			/// </summary>
			/// <param name="vector"></param>
			/// <returns></returns>
			static public Matrix<double>[] ToMatrices(Vector<double> vector, int d, int h, int w)
			{
				if (vector.Count != d * h * w) { throw new ArgumentException("vector size != d * h * w"); }
				Matrix<double>[] _mats = new Matrix<double>[d];
				double[,] _mat = new double[h, w];
				int _idx = 0;
				for (int i = 0; i < d; i++)
				{
					for (int j = 0; j < h; j++)
					{
						for (int k = 0; k < w; k++)
						{
							_mat[j, k] = vector[_idx];
							_idx++;
						}
					}
					_mats[i] = Matrix<double>.Build.DenseOfArray(_mat);
				}
				return (Matrix<double>[])_mats.Clone();
			}

			/// <summary>
			/// Vector → Matrix
			/// </summary>
			/// <param name="vector"></param>
			/// <returns></returns>
			static public Matrix<double> ToMatrix(Vector<double> vector, int h, int w)
			{
				if (vector.Count != h * w) { throw new ArgumentException("vector size != h * w"); }
				double[,] _mat = new double[h, w];
				int _idx = 0;
				for (int i = 0; i < h; i++)
				{
					for (int j = 0; j < w; j++)
					{
						_mat[i, j] = vector[_idx];
						_idx++;
					}
				}
				return Matrix<double>.Build.DenseOfArray(_mat);
			}
		}
	}
}
