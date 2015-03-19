using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace ConvolutionalNeuralNetworks
{
	namespace Layers
	{
		/// <summary>
		/// <para>MaxOut Layer Ver. FullyConnectedLayer</para>
		/// <para>なんかよくわからん</para>
		/// </summary>
		class MaxoutLayer : FullyConnectedLayer<Activations.Identity>
		{
			/// <summary>
			/// <para>Maxout処理をする前の内部ポテンシャル</para>
			/// <para>[in_size,out_size]</para>
			/// </summary>
			new Matrix<double> _outputs;

			/// <summary>
			/// バイアス
			/// </summary>
			new Matrix<double> _biases;

			public MaxoutLayer(int in_size, int out_size, string layer_name = "", Vector<double> weights = null, Vector<double> biases = null)
				: base(in_size, out_size, layer_name, weights, null)
			{
				// 内部ポテンシャル
				_outputs = Matrix<double>.Build.Dense(in_size, out_size, 0);

				// バイアス初期化
				if (biases != null && biases.Count == _in_size * _out_size) { _biases = Tools.Converters.ToMatrix(biases, _in_size, _out_size); }
				else { _biases = Matrix<double>.Build.Dense(_in_size, _out_size, 0); }
				_db = Matrix<double>.Build.Dense(_in_size, _out_size, 0);

				LayerType = "MaxoutLayer Ver.FCL";
				GenericsType = "Maxout";
			}

			public override void ForwardPropagation()
			{
				Parallel.For(0, _out_size, osz =>
				{
					for (int isz = 0; isz < _in_size; isz++)
					{
						_outputs[isz, osz] = _weights[isz, osz] * _inputs[isz] + _biases[isz, osz];
					}
				});
			}

			/// <summary>
			/// バイアス更新料 Δb
			/// </summary>
			new Matrix<double> _db;

			public override Vector<double> BackPropagation(Vector<double> next_delta)
			{
				var curt_delta = Matrix<double>.Build.Dense(_in_size, _out_size, 0);

				Parallel.For(0, _out_size, osz =>
				{
					curt_delta[_outputs.Column(osz).MaximumIndex(), osz] = next_delta[osz];
					for (int isz = 0; isz < _in_size; isz++)
					{
						_db[isz, osz] += curt_delta[isz, osz];
						_dw[isz, osz] += curt_delta[isz, osz] * _inputs[isz];
					}
				});

				return (curt_delta.PointwiseMultiply(_weights)).RowSums();
			}

			public override void WeightUpdate(double eta, double mu, double lambda)
			{
				// 更新
				// Δw(t) = -η∂E/∂w(t) + μΔw(t-1) - ηλw(t)
				var _dw_ = -eta * _dw + mu * _pre_dw - eta * lambda * _weights;
				_weights = _weights + _dw_;
				_biases -= eta * _db;

				// 正則化に使用する更新量保持
				_pre_dw = _dw_.Clone();

				_dw.Clear();
				_db.Clear();
			}

			public override Vector<double> Outputs
			{
				get
				{
					Vector<double> _tmp_outputs = Vector<double>.Build.Dense(_out_size);
					for (int osz = 0; osz < _out_size; osz++)
					{
						_tmp_outputs[osz] = _outputs.Column(osz).Maximum();
					}
					return _tmp_outputs.Clone();
				}
			}

			public override Vector<double> Biases
			{
				get
				{
					return Tools.Converters.ToVector(_biases);
				}
				protected set
				{
					if (_in_size * _out_size != value.Count) { throw new ArgumentException("Size of biases are different"); }
					_biases = Tools.Converters.ToMatrix(value, _in_size, _out_size);
				}
			}

			public override Vector<double> PredictOutputs
			{
				get { return this.Outputs; }
			}

			public override string ToString(string fmt = "o")
			{
				StringBuilder _res;
				switch (fmt)
				{
					case "b":
						_res = new StringBuilder("#Biases\n" + _wei_hei + "\t" + _wei_wid + "\n", _wei_size * 8);
						for (int bh = 0; bh < _wei_hei; bh++)
						{
							for (int bw = 0; bw < _wei_wid; bw++)
							{
								_res.Append(_biases[bh, bw] + "\t");
							}
							_res.Append("\n");
						}
						break;
					case "w":
						_res = new StringBuilder("#Weights\n" + _wei_hei + "\t" + _wei_wid + "\n", _wei_size * 8);
						for (int wh = 0; wh < _wei_hei; wh++)
						{
							for (int ww = 0; ww < _wei_wid; ww++)
							{
								_res.Append(_weights[wh, ww] + "\t");
							}
							_res.Append("\n");
						}
						break;
					case "i":
						_res = new StringBuilder("#Inputs\n" + _in_size + "\n", _in_size * 8);
						for (int isz = 0; isz < _in_size; isz++)
						{
							_res.Append(_inputs[isz] + "\t");
						}
						break;
					case "o":
						var _o = this.Outputs;
						_res = new StringBuilder("#Output\n" + _out_size + "\n", _out_size * 8);
						for (int osz = 0; osz < _out_size; osz++)
						{
							_res.Append(_activation.f(_o[osz]) + "\t");
						}
						break;
					case "l":
						_res = new StringBuilder("Inputs:" + _in_size + ", " + "Outputs:" + _out_size + ", " +
							"Weights:" + _wei_hei + "x" + _wei_wid + ", " +
							"Biases:" + _wei_hei + "x" + _wei_wid);
						break;
					default:
						_res = new StringBuilder("None\n");
						break;
				}

				return _res.ToString();
			}
		}
	}
}
