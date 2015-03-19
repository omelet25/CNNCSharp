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
		// DropOut and DropConnect

		/// <summary>
		/// DropOut Layer
		/// </summary>
		/// <typeparam name="ActivationType">活性化関数タイプ</typeparam>
		class DropOutLayer<ActivationType> : FullyConnectedLayer<ActivationType> where ActivationType : Activations.IActivation, new()
		{
			private MathNet.Numerics.Distributions.ContinuousUniform rand;

			/// <summary>
			/// DropOut する確率
			/// </summary>
			private double _drop_prob;

			/// <summary>
			/// DropOut するニューロンのマスク
			/// </summary>
			private int[] _drop_mask;

			public DropOutLayer(
				int in_size, int out_size, double drop_prob = 0.5, string layer_name = "",
				Vector<double> weights = null, Vector<double> biases = null)
				: base(in_size, out_size, layer_name, weights, biases)
			{
				// 擬似乱数器生成
				rand = new ContinuousUniform();

				// DropOut する確率
				if (drop_prob > 1.0 || drop_prob < 0)
				{
					Console.WriteLine("0 < drop_prob < 1.0. drop_prob is set 0.5");
					_drop_prob = 0.5;
				}
				else { _drop_prob = drop_prob; }

				// マスク初期化
				_drop_mask = new int[out_size];
				CreateDropMask();

				LayerType = "DropOutLayer";
				GenericsType = _activation.Type();
			}
			
			/// <summary>
			/// <para>順方向の計算</para>
			/// </summary>
			public override void ForwardPropagation()
			{
				base.ForwardPropagation();
			}

			public override Vector<double> BackPropagation(Vector<double> next_delta)
			{
				Vector<double> curt_delta = next_delta;
				//for (int osz = 0; osz < _out_size; osz++)
				Parallel.For(0, _out_size, osz =>
				{
					if (_drop_mask[osz] == 1)
					{
						// δ_[n,j] = next_delta_[n+1,j] * φ'(_outputs_[n,j]) * _drop_mask[j]
						curt_delta[osz] *= (_activation.df(_outputs[osz]) * _drop_mask[osz]);
						// Δb_[n,j] = (Σ w_[n+1] * δ_[n+1]) * φ'(_outputs_[n,j]) = curt_delta
						_db[osz] += curt_delta[osz];
						// Δw_[n] = δ_[n] * input_[n]
						for (int isz = 0; isz < _in_size; isz++) { _dw[isz, osz] += curt_delta[osz] * _inputs[isz]; }
					}
				});

				// return : Σ w_[n] * δ_[n]
				return (_weights * curt_delta).Clone();
			}

			public override void WeightUpdate(double eta, double mu, double lambda)
			{
				base.WeightUpdate(eta, mu, lambda);

				// Mini-batch sizeごとにマスクを初期化するため荷重更新時に行う
				CreateDropMask();
			}

			/// <summary>
			/// DropOutのmask生成
			/// </summary>
			private void CreateDropMask()
			{
				for (int i = 0; i < _drop_mask.Length; i++)
				{
					_drop_mask[i] = rand.Sample() < _drop_prob ? 0 : 1;
				}
			}

			/// <summary>
			/// <para>学習時のニューロン出力</para>
			/// <para>drop_maskでマスキングした出力</para>
			/// </summary>
			public override Vector<double> Outputs
			{
				get
				{
					return Vector<double>.Build.DenseOfEnumerable(
						_outputs.Select((val, idx) => { return _drop_mask[idx] == 1 ? _activation.f(val) : 0; })
					);
				}
			}

			/// <summary>
			/// <para>推定時のニューロン出力</para>
			/// <para>モデルの平均を取るため，DropOutする確率を掛ける</para>
			/// <para>ex.確率20%なら出力は80%にする</para>
			/// <para>f(_outputs) * (1.0 - prob)</para>
			/// </summary>
			public override Vector<double> PredictOutputs
			{
				get
				{
					return Vector<double>.Build.DenseOfEnumerable(
						_outputs.Select(val => _activation.f(val) * (1.0 - _drop_prob))
					);
				}
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
					case "lo":
						_res = new StringBuilder("#Learning Output\n" + _out_size + "\n", _out_size * 8);
						for (int osz = 0; osz < _out_size; osz++)
						{
							if (_drop_mask[osz] == 0) { _res.Append(0 + "\t"); }
							else
							{
								_res.Append(_activation.f(_outputs[osz]) + "\t");
							}
						}
						break;
					case "po":
						_res = new StringBuilder("#Predict Output\n" + _out_size + "\n", _out_size * 8);
						for (int osz = 0; osz < _out_size; osz++)
						{
							_res.Append((_activation.f(_outputs[osz]) * (1.0 - _drop_prob)) + "\t");
						}
						break;
					case "l":
						_res = new StringBuilder("Inputs:" + _in_size + ", " + "Outputs:" + _out_size + ", " +
							"Weights:" + _wei_hei + "x" + _wei_wid + ", " +
							"Biases:" + _out_size + ", " + "DropOutProb:" + _drop_prob);
						break;
					default:
						_res = new StringBuilder("None\n");
						break;
				}

				return _res.ToString();
			}
		}


		/// <summary>
		/// DropConnect Layer
		/// </summary>
		class DropConnectLayer<ActivationType> : FullyConnectedLayer<ActivationType> where ActivationType : Activations.IActivation, new()
		{
			private MathNet.Numerics.Distributions.ContinuousUniform rand;

			/// <summary>
			/// DropOut する確率
			/// </summary>
			private double _drop_prob;

			/// <summary>
			/// DropConnect する荷重接続のマスク
			/// </summary>
			private Matrix<double> _drop_mask;

			public DropConnectLayer(
				int in_size, int out_size, double drop_prob = 0.5, string layer_name = "",
				Vector<double> weights = null, Vector<double> biases = null)
				: base(in_size, out_size, layer_name, weights, biases)
			{
				rand = new ContinuousUniform();

				// DropConnect する確率
				if (drop_prob > 1.0 || drop_prob < 0)
				{
					Console.WriteLine("0 < drop_prob < 1.0. drop_prob is set 0.5");
					_drop_prob = 0.5;
				}
				else { _drop_prob = drop_prob; }

				// マスク初期化
				_drop_mask = Matrix<double>.Build.Dense(in_size, out_size, 0);
				CreateDropMask();

				LayerType = "DropConnectLayer";
				GenericsType = _activation.Type();
			}

			public override void ForwardPropagation()
			{
				// _outputs = (Mask*W)^T * _inputs
				_outputs = (_drop_mask.PointwiseMultiply(_weights)).TransposeThisAndMultiply(_inputs) + _biases;
			}

			public override Vector<double> BackPropagation(Vector<double> next_delta)
			{
				Vector<double> curt_delta = next_delta;
				//for (int osz = 0; osz < _out_size; osz++)
				Parallel.For(0, _out_size, osz =>
				{
					// δ_[n,j] = next_delta_[n+1,j] * φ'(_outputs_[n,j])
					curt_delta[osz] *= _activation.df(_outputs[osz]);
					// Δb_[n,j] = (Σ w_[n+1] * δ_[n+1]) * φ'(_outputs_[n,j]) = curt_delta
					_db[osz] += curt_delta[osz];
					// Δw_[n] = δ_[n] * input_[n]
					for (int isz = 0; isz < _in_size; isz++) { _dw[isz, osz] += curt_delta[osz] * _inputs[isz]; }
				});

				// return : Σ w_[n] * δ_[n]
				return (_weights * curt_delta).Clone();
			}

			public override Vector<double> PredictOutputs
			{
				get
				{
					return Vector<double>.Build.DenseOfEnumerable(
						(_weights.TransposeThisAndMultiply(_inputs) + _biases).Select(val =>
						{
							return _activation.f(val) * (1.0 - _drop_prob);
						})
					);
				}
			}

			public override void WeightUpdate(double eta,double mu,double lambda)
			{
				base.WeightUpdate(eta, mu, lambda);

				// Mini-batch sizeごとにマスクを初期化するため荷重更新時に行う
				CreateDropMask();
			}
			
			/// <summary>
			/// DropConnectのmask生成
			/// </summary>
			private void CreateDropMask()
			{
				for (int isz = 0; isz < _in_size; isz++)
				{
					for (int osz = 0; osz < _out_size; osz++)
					{
						_drop_mask[isz, osz] = rand.Sample() < _drop_prob ? 0 : 1;
					}
				}
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
					case "lo":
						_res = new StringBuilder("#Learning Output\n" + _out_size + "\n", _out_size * 8);
						for (int osz = 0; osz < _out_size; osz++)
						{
							_res.Append(_activation.f(_outputs[osz]) + "\t");
						}
						break;
					case "po":
						var _o = this.PredictOutputs;
						_res = new StringBuilder("#Predict Output\n" + _out_size + "\n", _out_size * 8);
						for (int osz = 0; osz < _out_size; osz++)
						{
							_res.Append(_o[osz] + "\t");
						}
						break;
					case "l":
						_res = new StringBuilder("Inputs:" + _in_size + ", " + "Outputs:" + _out_size + ", " +
							"Weights:" + _wei_hei + "x" + _wei_wid + ", " +
							"Biases:" + _out_size + ", " + "DropConnectProb:" + _drop_prob);
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
