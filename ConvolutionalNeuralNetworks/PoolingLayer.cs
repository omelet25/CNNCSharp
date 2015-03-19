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
		/// <para>Pooling Layer</para>
		/// </summary>
		class PoolingLayer<PoolingType> : BaseLayer
			where PoolingType : Poolings.IPooling, new()
		{
			/// <summary>
			/// 入力
			/// </summary>
			private Matrix<double>[] _inputs;
			/// <summary>
			/// 出力
			/// </summary>
			private Matrix<double>[] _outputs;

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
			/// Pooling のサイズ
			/// </summary>
			private int _pool_size;

			/// <summary>
			/// Pooling関数
			/// </summary>
			PoolingType _pooling = new PoolingType();

			/// <summary>Pooling Layer</summary>
			/// <param name="in_height">入力高さ</param>
			/// <param name="in_width">入力幅</param>
			/// <param name="in_depth">入力深さ</param>
			/// <param name="pooling_size">pooling window size</param>
			/// <param name="stride">フィルタ移動度</param>
			/// <param name="layer_name">レイヤー名</param>
			public PoolingLayer(
				int in_height, int in_width, int in_depth,
				int pooling_size = 2, int stride = 2, string layer_name = "")
				: base(layer_name)
			{
				// 入力
				_in_hei = in_height; _in_wid = in_width; _in_dep = in_depth;
				_in_size = in_height * in_width * in_depth;
				_inputs = new Matrix<double>[_in_dep];
				for (int i = 0; i < _in_dep; i++)
				{
					_inputs[i] = Matrix<double>.Build.Dense(_in_hei, _in_wid, 0);
				}

				_pool_size = pooling_size;

				// 出力
				_out_hei = (_in_hei - _pool_size) / stride + 1;
				_out_wid = (_in_wid - _pool_size) / stride + 1;
				_out_dep = _in_dep;
				_out_size = _out_hei * _out_wid * _out_dep;

				_outputs = new Matrix<double>[_out_dep];
				for (int i = 0; i < _out_dep; i++)
				{
					_outputs[i] = Matrix<double>.Build.Dense(_out_hei, _out_wid, 0);
				}

				LayerType = "PoolingLayer";
				GenericsType = _pooling.Type();

				_stride = stride;
			}

			public override void ForwardPropagation() { Pooling(); }

			/// <summary>
			/// BP (Pooling ver.)
			/// </summary>
			/// <param name="next_delta">
			/// 次層から来た誤差信号 (Σ δ_[l+1] * w_[l+1])
			/// </param>
			/// <returns>前層へ伝播する誤差信号 (Σ δ_[l] * w_[l])</returns>
			public override Vector<double> BackPropagation(Vector<double> next_delta)
			{
				// Vector<double>[_outd * _outh * _outw] => Matrix<double>[_outd][_outh,_outw]にする
				Matrix<double>[] _curt_delta = Tools.Converters.ToMatrices(next_delta, _out_dep, _out_hei, _out_wid);

				// 前層へ伝播する誤差 
				Matrix<double>[] prev_delta = new Matrix<double>[_in_dep];

				//for (int id = 0; id < _in_dep; id++)
				Parallel.For(0, _in_dep, id =>
				{
					prev_delta[id] = Matrix<double>.Build.Dense(_in_hei, _in_wid, 0);

					int ih = 0;
					for (int oh = 0; oh < _out_hei; oh++)
					{
						int iw = 0;
						for (int ow = 0; ow < _out_wid; ow++)
						{
							// (Σ δ * w) * φ'(_outputs)
							var _df_delta = _pooling.df(_inputs[id].SubMatrix(ih, _pool_size, iw, _pool_size)) * _curt_delta[id][oh, ow];

							// inv_f_delta 対応する pre_delta の場所に加算
							for (int u = 0; u < _pool_size; u++)
							{
								for (int v = 0; v < _pool_size; v++)
								{
									prev_delta[id][ih + u, iw + v] += _df_delta[u, v];
								}
							}
							iw += _stride;
						}
						ih += _stride;
					}
				});

				return Tools.Converters.ToVector(prev_delta, _in_hei, _in_wid);
			}

			/// <summary>
			/// <para>Pooling</para>
			/// </summary>
			private void Pooling()
			{
				//for (int od = 0; od < _out_dep; id++)
				Parallel.For(0, _out_dep, od =>
				{
					int ih = 0;
					for (int oh = 0; oh < _out_hei; oh++)
					{
						int iw = 0;
						for (int ow = 0; ow < _out_wid; ow++)
						{
							_outputs[od][oh, ow] = _pooling.f(_inputs[od].SubMatrix(ih, _pool_size, iw, _pool_size));
							iw += _stride;
						}
						ih += _stride;
					}
				});
			}
			
			public override string ToString(string fmt = "o")
			{
				StringBuilder _res;
				switch (fmt)
				{
					case "i":
						_res = new StringBuilder("#Inputs\n" + _in_dep + "\t" + _in_hei + "\t" + _in_wid + "\n", _in_size * 8);
						for (int id = 0; id < _in_dep; id++)
						{
							for (int ih = 0; ih < _in_hei; ih++)
							{
								for (int iw = 0; iw < _in_wid; iw++)
								{
									_res.Append(_inputs[id][ih,iw] + "\t");
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
									_res.Append(_outputs[od][oh, ow] + "\t");
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
							"PoolingSize:" + _pool_size + "x" + _pool_size + ", " + "Stride:" + _stride);
						break;
					default:
						_res = new StringBuilder("None\n");						
						break;
				}

				return _res.ToString();
			}

			public override Vector<double> Inputs
			{
				set
				{
					if (_in_size != value.Count) { throw new ArgumentException("Size of inputs is different"); }
					_inputs = Tools.Converters.ToMatrices(value, _in_dep, _in_hei, _in_wid);
				}
			}

			public override Vector<double> Outputs
			{
				get { return Tools.Converters.ToVector(_outputs, _out_hei, _out_wid); }
			}
		}
	}
}