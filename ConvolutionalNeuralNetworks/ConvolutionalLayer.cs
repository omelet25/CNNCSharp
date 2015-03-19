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
		/// <para>畳み込み層</para>
		/// <typeparam name="ActivationType">
		/// <para>活性化関数タイプ</para>
		/// <para>Tools.Actionvations内のクラス</para>
		/// </typeparam>
		/// </summary>
		class ConvolutionalLayer<ActivationType> : BaseLayer
			where ActivationType : Activations.IActivation, new()
		{
			/// <summary>
			/// 入力
			/// </summary>
			protected Matrix<double>[] _inputs;
			/// <summary>
			/// 出力（活性化関数通ってない）
			/// </summary>
			protected Matrix<double>[] _outputs;
			/// <summary>
			/// 畳み込みフィルタ
			/// </summary>
			protected Matrix<double>[] _kernels;
			/// <summary>
			/// バイアス
			/// </summary>
			protected Vector<double> _biases;

			/// <summary>
			/// 活性化関数
			/// </summary>
			protected ActivationType _activation = new ActivationType();

			/// <summary>
			/// <para>入力関連のパラメータ : _in_[hei,wid,dep]</para>
			/// <para>hei	: 行</para>
			/// <para>wid	: 列</para>
			/// <para>dep	: 深さ</para>
			/// </summary>
			private int _in_hei, _in_wid, _in_dep;

			/// <summary>
			/// <para>出力関連のパラメータ : _out_[hei,wid,dep]</para>
			/// <para>hei	: 行</para>
			/// <para>wid	: 列</para>
			/// <para>dep	: 深さ</para>
			/// </summary>
			protected int _out_hei, _out_wid, _out_dep;

			/// <summary>
			/// <para>荷重，フィルタ関連のパラメータ : _wei_[hei,wid,dep,size]</para>
			/// <para>hei	: 行</para>
			/// <para>wid	: 列</para>
			/// <para>dep	: 深さ</para>
			/// <para>size	: 総数</para>
			/// </summary>
			protected int _wei_hei, _wei_wid, _wei_dep, _wei_size;

			/// <summary>
			/// padding
			/// </summary>
			protected int _padding;

			/// <summary>
			/// padding を考慮に入れた入力サイズ
			/// </summary>
			protected int _in_size_pad, _in_hei_pad, _in_wid_pad;

			/// <summary>
			/// connection_table
			/// </summary>
			protected int[,] _connection_table;
			
			/// <summary>
			/// Convolutional Layer
			/// </summary>
			/// <param name="in_height">入力高さ</param>
			/// <param name="in_width">入力幅</param>
			/// <param name="in_depth">入力深さ</param>
			/// <param name="kernel_size">畳み込みカーネルサイズ</param>
			/// <param name="out_depth">出力深さ</param>
			/// <param name="stride">カーネル移動量</param>
			/// <param name="padding">padding</param>
			/// <param name="connection_table">connection_table[in_depth,out_depth] = 0 or 1，null or サイズが違う場合全結合</param>
			/// <param name="eta">学習係数</param>
			/// <param name="layer_name">レイヤー名</param>
			/// <param name="kernels"><para>初期畳み込みカーネル</para>
			/// <para>null or サイズが違う場合はrand(-1.0,1.0) / sqrt(in_depth * kernel_size^2)で初期化</para>
			/// </param>
			/// <param name="biases">初期バイアス，null or サイズが違う場合は 0 で初期化</param>
			public ConvolutionalLayer(
				int in_height, int in_width, int in_depth, int kernel_size = 3, int out_depth = 32,
				int stride = 1, int padding = 0, int[] connection_table = null, string layer_name = "",
				Vector<double> kernels = null, Vector<double> biases = null)
				: base(layer_name)
			{
				// 入力 no padding
				_in_hei = in_height; _in_wid = in_width; _in_dep = in_depth;
				_in_size = in_height * in_width * in_depth;

				// 入力 with padding
				_in_hei_pad = in_height + padding * 2; _in_wid_pad = in_width + padding * 2;
				_in_size_pad = _in_hei_pad * _in_wid_pad * _in_dep;

				// padding を考慮して入力生成
				_inputs = new Matrix<double>[_in_dep];
				for (int id = 0; id < _in_dep; id++)
				{
					_inputs[id] = Matrix<double>.Build.Dense(_in_hei_pad, _in_wid_pad, 0);
				}

				// 畳み込みカーネル
				// 深さのアクセスは id * out_depth + od
				// id : in_depthのループ
				// od : out_depthのアクセス
				_wei_hei = _wei_wid = kernel_size; _wei_dep = in_depth * out_depth;
				_wei_size = kernel_size * kernel_size * _wei_dep;

				// 出力
				_out_hei = (_in_hei_pad - _wei_hei) / stride + 1;
				_out_wid = (_in_wid_pad - _wei_wid) / stride + 1;
				_out_dep = out_depth;
				_out_size = _out_hei * _out_wid * _out_dep;


				LayerType = "ConvolutionalLayer";
				GenericsType = _activation.Type();

				_stride = stride;
				_padding = padding;

				// 出力
				_outputs = new Matrix<double>[_out_dep];
				for (int od = 0; od < _out_dep; od++)
				{
					_outputs[od] = Matrix<double>.Build.Dense(_out_hei, _out_wid, 0);
				}

				// connection tableの初期化
				_connection_table = new int[in_depth, out_depth];
				if (connection_table != null && connection_table.Length == _wei_dep)
				{
					for (int id = 0; id < in_depth; id++)
					{
						for (int od = 0; od < out_depth; od++)
						{
							_connection_table[id, od] = connection_table[id * out_depth + od];
						}
					}
				}
				else
				{
					for (int id = 0; id < in_depth; id++)
					{
						for (int od = 0; od < out_depth; od++)
						{
							_connection_table[id, od] = 1;
						}
					}
				}

				// 畳み込みカーネル
				// 深さのアクセスは id * out_depth + od
				// id : in_depthのループ
				// od : out_depthのアクセス
				// 初期畳み込みカーネル生成
				_kernels = new Matrix<double>[_wei_dep];
				if (kernels != null && kernels.Count == _wei_size) { GenerateWeights(kernels); }
				else
				{
					var w_bd = Math.Sqrt(_in_dep * _wei_hei * _wei_wid);
					GenerateWeights(-1.0 / w_bd, 1.0 / w_bd);
				}

				// 初期バイアス生成
				if (biases != null && biases.Count == out_depth) { _biases = biases.Clone(); }
				else { _biases = Vector<double>.Build.Dense(out_depth, 0); }

				// 各荷重(カーネル)の更新量 Δw
				_dw = new Matrix<double>[_wei_dep];
				_pre_dw = new Matrix<double>[_wei_dep];
				for (int wd = 0; wd < _wei_dep; wd++)
				{
					_dw[wd] = Matrix<double>.Build.Dense(_wei_hei, _wei_wid, 0);
					_pre_dw[wd] = Matrix<double>.Build.Dense(_wei_hei, _wei_wid, 0);
				}
				// バイアスの更新量 Δb
				_db = Vector<double>.Build.Dense(out_depth, 0);

			}

			/// <summary>
			/// 畳み込みカーネルを一様乱数で生成
			/// </summary>
			public override void GenerateWeights(double lower = -0.1, double upper = 0.1)
			{
				for (int id = 0; id < _in_dep; id++)
				{
					for (int od = 0; od < _out_dep; od++)
					{
						int wd = id * _out_dep + od;
						if (_connection_table[id, od] == 0)
						{
							_kernels[wd] = Matrix<double>.Build.Dense(_wei_hei, _wei_wid, 0);
						}
						else
						{
							_kernels[wd] = Matrix<double>.Build.Random(_wei_hei, _wei_wid, new ContinuousUniform(lower, upper));
						}
					}
				}
			}
			/// <summary>
			/// 荷重を任意に生成
			/// </summary>
			/// <param name="filters"></param>
			/// <returns></returns>
			public override void GenerateWeights(Vector<double> weights)
			{
				Weights = weights;
			}

			/// <summary>
			/// 畳み込み
			/// </summary>
			/// <returns>畳み込み成功</returns>
			public override void ForwardPropagation() { Convolution(); }

			/// <summary>
			/// 各荷重(カーネル)の更新量 Δw
			/// </summary>
			protected Matrix<double>[] _dw;
			/// <summary>
			/// バイアスの更新量 Δw
			/// </summary>
			protected Vector<double> _db;

			/// <summary>
			/// <para>前epochでの荷重の更新量 Δw(t-1)</para>
			/// <para>weight decay and momentum で使用</para>
			/// </summary>
			protected Matrix<double>[] _pre_dw;

			/// <summary>
			/// 誤差逆伝播法(Conv Layer ver.)
			/// </summary>
			/// <param name="next_delta">
			/// 次層から来た誤差信号 (Σ δ_[l+1] * w_[l+1])
			/// </param>
			/// <returns>前層へ伝播する誤差信号 (Σ δ_[l] * w_[l])</returns>
			public override Vector<double> BackPropagation(Vector<double> next_delta)
			{
				// Vector<double>[_outd * _outh * _outw] => Matrix<double>[_outd][_outh,_outw]にする
				Matrix<double>[] _curt_delta = Tools.Converters.ToMatrices(next_delta, _out_dep, _out_hei, _out_wid);

				// 各荷重(フィルタ)の更新量 Δw の計算
				// 1. δ_[l,j] = next_delta_[k] * φ'(_outputs_[l,j])
				// 2. Δw_[l,j] = Σ δ_[l,j] * _output_[l-1,j] = Σ δ_[l,j] * _inputs_[l,j]
				//for (int od = 0; od < _out_dep; od++)
				Parallel.For(0, _out_dep, od =>
				{
					// バイアスの更新量 Δb = Σ δ_[l,j]
					double _db_sum = 0;

					int _ih = 0;
					for (int oh = 0; oh < _out_hei; oh++)
					{
						int _iw = 0;
						for (int ow = 0; ow < _out_wid; ow++)
						{
							// 1. δ_[l,j] = next_delta_[k] * φ'(_outputs_[l,j])
							_curt_delta[od][oh, ow] *= _activation.df(_outputs[od][oh, ow]);

							// Δb = Σ δ_[l,j]
							_db_sum += _curt_delta[od][oh, ow];

							for (int id = 0; id < _in_dep; id++)
							{
								// connection_table
								if (_connection_table[id, od] == 1)
								{
									// 2.  Δw_[l,j] = Σ δ_[l,j] * _output_[l-1,j] = Σ δ_[l,j] * _inputs_[l,j]
									_dw[od] += _curt_delta[od][oh, ow] * _inputs[id].SubMatrix(_ih, _wei_hei, _iw, _wei_wid);
								}
							}
							_iw += _stride;
						}
						_ih += _stride;
					}

					// バイアスの更新量 Δb = Σ δ_[l,j]
					_db[od] += _db_sum;
				});

				// 前層の誤差計算で使う (Σ δ_[l] * w_[l]) の計算
				// 実際に前層に渡す誤差信号
				Matrix<double>[] prev_delta = new Matrix<double>[_in_dep];

				//for (int id = 0; id < _in_dep; id++)
				Parallel.For(0, _in_dep, id =>
				{
					prev_delta[id] = Matrix<double>.Build.Dense(_in_hei, _in_wid);

					// padding 込の計算
					var _tmp_delta = Matrix<double>.Build.Dense(_in_hei_pad, _in_wid_pad);

					int _ih = 0;
					for (int oh = 0; oh < _out_hei; oh++)
					{
						int _iw = 0;
						for (int ow = 0; ow < _out_wid; ow++)
						{
							// curt_delta[l,j] = next_delta_[k] * φ'(_outputs_[l,j])
							// (Σ δ_[l] * w_[l]) = (Σ curt_delta[l,j] * kernel_[l])
							// _wo は窓サイズ
							var _wdelta = Matrix<double>.Build.Dense(_wei_hei, _wei_wid, 0);
							for (int od = 0; od < _out_dep; od++)
							{
								if (_connection_table[id, od] == 1)
								{
									_wdelta += _kernels[id * _out_dep + od] * _curt_delta[od][oh, ow];
								}
							}

							// _wdelta_[l] を対応する prev_delta_padd に加算
							for (int u = 0; u < _wei_hei; u++)
							{
								for (int v = 0; v < _wei_wid; v++)
								{
									_tmp_delta[_ih + u, _iw + v] += _wdelta[u, v];
								}
							}
							_iw += _stride;
						}
						_ih += _stride;
					}

					// padding を除いた状態にする
					prev_delta[id] = _tmp_delta.SubMatrix(_padding, _in_hei, _padding, _in_wid);
				});

				// 前層へ誤差信号伝達
				return Tools.Converters.ToVector(prev_delta, _in_hei, _in_wid);
			}

			/// <summary>
			/// 荷重更新
			/// </summary>
			public override void WeightUpdate(double eta, double mu, double lambda)
			{
				for (int wd = 0; wd < _wei_dep; wd++)
				{
					// Δw(t) = -η∂E/∂w(t) + μΔw(t-1) - ηλw(t)
					var _dw_ = -eta * _dw[wd] + mu * _pre_dw[wd] - eta * lambda * _kernels[wd];
					// w(t) = w(t-1) + Δw(t)
					_kernels[wd] = _kernels[wd] + _dw_;
					
					// 荷重更新量保持
					_pre_dw[wd] = _dw_.Clone();

					_dw[wd].Clear();
				}
				// b(t+1) = b(t) - ηΔb(t+1)
				_biases -= eta * _db;

				_db.Clear();
			}

			/// <summary>
			/// <para>畳み込み</para>
			/// <para>_outputs[k] = Σ_{j∈L_inputs} _inputs[j] * kernel[k]</para>
			/// </summary>
			protected virtual void Convolution()
			{
				//for (int od = 0; od < _out_dep; od++)
				Parallel.For(0, _out_dep, od =>
				{
					var _tmp_out = Matrix<double>.Build.Dense(_out_hei, _out_wid, 0);

					for (int id = 0; id < _in_dep; id++)
					{
						// 結合されているかどうか
						if (_connection_table[id, od] == 1)
						{
							int ih = 0;
							for (int oh = 0; oh < _out_hei; oh++)
							{
								int iw = 0;
								for (int ow = 0; ow < _out_wid; ow++)
								{
									_tmp_out[oh, ow] += _inputs[id].SubMatrix(ih, _wei_hei, iw, _wei_wid).PointwiseMultiply(_kernels[id * _out_dep + od]).Enumerate().Sum();
									iw += _stride;
								}
								ih += _stride;
							}
						}
					}
					_outputs[od] = _tmp_out + _biases[od];
				});
			}

			/// <summary>
			/// <para>入力代入</para>
			/// <para>代入の際 padding を考慮に入れる</para>
			/// </summary>
			public override Vector<double> Inputs
			{
				set
				{
					if (_in_size != value.Count) { throw new ArgumentException("Size of inputs is different"); }
					int _idx = 0;
					for (int id = 0; id < _in_dep; id++)
					{
						for (int ih = 0; ih < _in_hei; ih++)
						{
							for (int iw = 0; iw < _in_wid; iw++)
							{
								_inputs[id][ih + _padding, iw + _padding] = value[_idx];
								_idx++;
							}
						}
					}
				}
			}

			/// <summary>
			/// Outputs = activation_f(_outputs)
			/// </summary>
			public override Vector<double> Outputs
			{
				get
				{
					return Tools.Converters.ToVector(_outputs.Select(_ => _.Map(_activation.f)).ToArray(), _out_hei, _out_wid);
				}
			}

			/// <summary>
			/// カーネル（荷重）
			/// </summary>
			public override Vector<double> Weights
			{
				get { return Tools.Converters.ToVector(_kernels, _wei_hei, _wei_wid); }
				protected set
				{
					if (_wei_size != value.Count) { throw new ArgumentException("Size of kernels are different"); }
					_kernels = Tools.Converters.ToMatrices(value, _wei_dep, _wei_hei, _wei_wid);
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
					if (_out_dep != value.Count) { throw new ArgumentException("Size of biases are different"); }
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
						_res = new StringBuilder("#Kernels\n" + _wei_dep + "\t" + _wei_hei + "\t" + _wei_wid + "\n", _wei_size * 8);
						for (int wd = 0; wd < _wei_dep; wd++)
						{
							for (int wh = 0; wh < _wei_hei; wh++)
							{
								for (int ww = 0; ww < _wei_wid; ww++)
								{
									_res.Append(_kernels[wd][wh, ww] + "\t");
								}
								_res.Append("\n");
							}
							_res.Append("\n");
						}
						break;
					case "i":
						_res = new StringBuilder("#Inputs\n" + _in_dep + "\t" + _in_hei + "\t" + _in_wid + "\n", _in_size * 8);
						for (int id = 0; id < _in_dep; id++)
						{
							for (int ih = 0; ih < _in_hei; ih++)
							{
								for (int iw = 0; iw < _in_wid; iw++)
								{
									_res.Append(_inputs[id][_padding + ih, _padding + iw] + "\t");
								}
								_res.Append("\n");
							}
							_res.Append("\n");
						}
						break;
					case "p":
						_res = new StringBuilder(
							"#Inputs with padding:" + _padding + "\n" +
							_in_dep + "\t" + _in_hei_pad + "\t" + _in_wid_pad + "\n", _in_size_pad * 8);
						for (int id = 0; id < _in_dep; id++)
						{
							for (int ihp = 0; ihp < _in_hei_pad; ihp++)
							{
								for (int iwp = 0; iwp < _in_wid_pad; iwp++)
								{
									_res.Append(_inputs[id][ihp, iwp] + "\t");
								}
								_res.Append("\n");
							}
							_res.Append("\n");
						}
						break;
					case "o":
						_res = new StringBuilder("#Output\n" + _out_dep + "\t" + _out_hei + "\t" + _out_wid + "\n", _out_size * 8);
						for (int od = 0; od < _out_dep; od++)
						{
							for (int oh = 0; oh < _out_hei; oh++)
							{
								for (int ow = 0; ow < _out_wid; ow++)
								{
									_res.Append(_activation.f(_outputs[od][oh, ow]) + "\t");
								}
								_res.Append("\n");
							}
							_res.Append("\n");
						}
						break;
					case "l":
						_res = new StringBuilder(
							"Inputs:" + _in_hei + "x" + _in_wid + "x" + _in_dep + ", " +
							"Outputs:" + _out_hei + "x" + _out_wid + "x" + _out_dep + ", " +
							"Kernels:" + _in_dep + "x" + _out_dep + "x" + _wei_hei + "x" + _wei_wid + ", " +
							"Biases:" + _out_dep + "," + "Stride:" + _stride + ", " + "Padding:" + _padding);
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