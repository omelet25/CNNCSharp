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
		/// FullyConnectedLayer
		/// </summary>
		/// <typeparam name="ActivationType"></typeparam>
		class FullyConnectedLayer<ActivationType> : BaseLayer
			where ActivationType : Activations.IActivation, new()
		{
			/// <summary>
			/// 活性化関数
			/// </summary>
			protected ActivationType _activation = new ActivationType();

			/// <summary>
			/// <para>入力</para>
			/// <para>input_this = output_prev = activation_f(Σ w_prev * input_prev)</para>
			/// <para>前層の出力と等しい</para>
			/// </summary>
			protected Vector<double> _inputs;
			/// <summary>
			/// <para>出力</para>
			/// <para>output_this = Σ w_this * input_this</para>
			/// </summary>
			protected Vector<double> _outputs;

			/// <summary>
			/// <para>荷重</para>
			/// <para>[in_num][out_num]</para>
			/// <para>計算で使うときは転置</para>
			/// </summary>
			protected Matrix<double> _weights;

			/// <summary>
			/// バイアス
			/// </summary>
			protected Vector<double> _biases;

			/// <summary>
			/// <para>荷重，フィルタ関連のパラメータ : _wei_[hei,wid,dep,size]</para>
			/// <para>hei	: 行</para>
			/// <para>wid	: 列</para>
			/// <para>dep	: 深さ</para>
			///// <para>size	: 総数</para>
			/// </summary>
			protected int _wei_hei, _wei_wid, _wei_size;

			/// <summary>
			/// コンストラクタ
			/// </summary>
			/// <param name="in_size">入力数</param>
			/// <param name="out_size">出力数</param>
			/// <param name="eta">学習係数</param>
			/// <param name="layer_name">レイヤー名</param>
			/// <param name="weights">初期荷重，null or sizeが違う場合 rand(-1.0,1.0) / sqrt(in_size) で初期化</param>
			/// <param name="biases">初期バイアス，null or sizeが違う場合 0</param>
			public FullyConnectedLayer(int in_size, int out_size, string layer_name = "", Vector<double> weights = null, Vector<double> biases = null)
				: base(layer_name)
			{
				_in_size = in_size; _out_size = out_size;
				_inputs = Vector<double>.Build.Dense(_in_size);
				_outputs = Vector<double>.Build.Dense(_out_size);

				// 荷重
				_wei_hei = _in_size; _wei_wid = _out_size; _wei_size = _in_size * _out_size;
				// 荷重初期化
				if (weights != null && weights.Count == _wei_size) { GenerateWeights(weights); }
				else { var _w_bd = in_size; GenerateWeights(-1.0 / _w_bd, 1.0 / _w_bd); }
				_dw = Matrix<double>.Build.Dense(_wei_hei, _wei_wid, 0);
				_pre_dw = Matrix<double>.Build.Dense(_wei_hei, _wei_wid, 0);

				// バイアス初期化
				if (biases != null && biases.Count == _out_size) { _biases = biases.Clone(); }
				else { _biases = Vector<double>.Build.Dense(_out_size, 0); }
				_db = Vector<double>.Build.Dense(_out_size, 0);

				LayerType = "FullyConnectedLayer";
				GenericsType = _activation.Type();
			}

			public override void ForwardPropagation()
			{
				// _outputs =  w^T * input
				_outputs = _weights.TransposeThisAndMultiply(_inputs) + _biases;
			}

			/// <summary>
			/// 荷重の更新量 Δw
			/// </summary>
			protected Matrix<double> _dw;
			/// <summary>
			/// バイアスの更新量 Δb
			/// </summary>
			protected Vector<double> _db;
			/// <summary>
			/// 前回の荷重更新量 Δw(t-1)
			/// </summary>
			protected Matrix<double> _pre_dw;

			/// <summary>
			/// <para>BP</para>
			/// </summary>
			/// <param name="next_delta">
			/// <para>wdelta_[n+1]= w_[n+1] * δ_[n+1]</para>
			/// </param>
			/// <returns>w_[n] * δ_[n]</returns>
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

			/// <summary>
			/// <para>荷重更新</para>
			/// <para>伝播した誤差をリセット</para>
			/// </summary>
			public override void WeightUpdate(double eta,double mu,double lambda)
			{
				// Δw(t) = -η∂E/∂w(t) + μΔw(t-1) - ηλw(t)
				var _dw_ = -eta * _dw + mu * _pre_dw - eta * lambda * _weights;
				_weights = _weights + _dw_;

				// b_[n] = b_[n] - ηΔb_[n]
				_biases -= eta * _db;

				// 正則化に使う更新量を保持
				_pre_dw = _dw_.Clone();

				// 初期化
				_dw.Clear();
				_db.Clear();
			}

			/// <summary>
			/// 荷重を一様乱数で生成
			/// </summary>
			public override void GenerateWeights(double lower = -0.1, double upper = 0.1)
			{
				_weights = Matrix<double>.Build.Random(_wei_hei, _wei_wid, new ContinuousUniform(lower, upper));
			}
			/// <summary>
			/// 荷重を任意に生成
			/// </summary>
			/// <param name="weights"></param>
			/// <returns></returns>
			public override void GenerateWeights(Vector<double> weights) { Weights = weights; }

			public override Vector<double> Inputs
			{
				set
				{
					if (_in_size != value.Count) { throw new ArgumentException("Size of inputs is different"); }
					_inputs = value.Clone();
				}
			}

			/// <summary>
			/// Output = activation_f(this.outputs)
			/// </summary>
			public override Vector<double> Outputs { get { return _outputs.Map(_activation.f); } }

			public override Vector<double> Weights
			{
				get { return Tools.Converters.ToVector(_weights); }
				protected set
				{
					if (_wei_hei * _wei_wid != value.Count) { throw new ArgumentException("Size of weights is different"); }
					_weights = Tools.Converters.ToMatrix(value, _wei_hei, _wei_wid);
				}
			}

			/// <summary>
			/// バイアス
			/// </summary>
			public override Vector<double> Biases
			{
				get
				{
					return _biases.Clone();
				}
				protected set
				{
					if (_out_size != value.Count) { throw new ArgumentException("Size of biases are different"); }
					_biases = value.Clone();
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
						_res = new StringBuilder("#Inputs\n" + _in_size + "\n",_in_size * 8);
						for (int isz = 0; isz < _in_size; isz++)
						{
							_res.Append(_inputs[isz] + "\t");
						}
						break;
					case "o":
						_res = new StringBuilder("#Output\n" + _out_size + "\n", _out_size * 8);
						for (int osz = 0; osz < _out_size; osz++)
						{
							_res.Append( _activation.f(_outputs[osz]) + "\t");
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
