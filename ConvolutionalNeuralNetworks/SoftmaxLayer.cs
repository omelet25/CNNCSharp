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
		/// <para>Softmax Layer</para>
		/// <para>よくわからん・・・</para>
		/// </summary>
		class SoftmaxLayer : FullyConnectedLayer<Activations.Identity>
		{
			/// <summary>
			/// 誤差が発散しないよう e^{x_i} の値を制限
			/// </summary>
			const double _ymin = 1.0e-10;
			Func<double, double> _obd_f = new Func<double, double>(_val => { return _val < _ymin ? _ymin : _val; });

			public SoftmaxLayer(int in_size, int out_size, string layer_name = "", Vector<double> weights = null, Vector<double> biases = null)
				: base(in_size, out_size, layer_name, weights, biases)
			{
				LayerType = "SoftmaxLayer";
				GenericsType = "Softmax";
			}

			public override void ForwardPropagation()
			{
				// _e_[i] = Σ w_[j,i] * x_[i] + b_[i]
				var _e = _weights.TransposeThisAndMultiply(_inputs) + _biases;
				// 指数関数が発散しないように
				// _e_[i] = exp[(_e_[i] - max(_e))]
				_e = (_e - _e.Maximum()).PointwiseExp().Map(_obd_f);
				// _outputs_[i] = _e_[i] / Σ _e
				_outputs = (_e / _e.Sum());
			}

			public override Vector<double> BackPropagation(Vector<double> next_delta)
			{
				var curt_delta = next_delta.Clone();

				Parallel.For(0, _out_size, osz =>
				{
					// δ_[n,j] = curt_delta_[n+1,j] * φ' = curt_delta_[n+1,j] * _outputs[j] * (1.0 - _outputs[j])
					curt_delta[osz] *= (_outputs[osz] * (1.0 - _outputs[osz]));
					// Δb_[n,j] = δ_[n,j]
					_db[osz] += curt_delta[osz];
					// Δw_[n] = δ_[n] * _inputs_[n]
					for (int isz = 0; isz < _in_size; isz++)
					{
						_dw[isz, osz] += curt_delta[osz] * _inputs[isz];
					}
				});

				// return : Σ w_[n] * δ_[n]
				return (_weights * curt_delta).Clone();
			}

			public override Vector<double> Outputs
			{
				get { return _outputs.Clone(); }
			}

			public override string ToString(string fmt = "o")
			{
				StringBuilder _res;
				switch (fmt)
				{
					case "b":
						_res = new StringBuilder("#Biases\n" + _biases.Count + "\n", _biases.Count * 8);
						for (int bsz = 0; bsz < _biases.Count; bsz++)
						{
							_res.Append(_biases[bsz] + "\t");
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
						_res = new StringBuilder("#Output\n" + _out_size + "\n", _out_size * 8);
						for (int osz = 0; osz < _out_size; osz++)
						{
							_res.Append(_outputs[osz] + "\t");
						}
						break;
					case "l":
						_res = new StringBuilder("Inputs:" + _in_size + ", " + "Outputs:" + _out_size + ", " +
							"Weights:" + _wei_hei + "x" + _wei_wid + ", " +
							"Biases:" + _out_size);
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