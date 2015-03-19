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
		/// レイヤーごとの処理
		/// </summary>
		/// <typeparam name="ElementWiseType"></typeparam>
		class ElemWiseLayer<ElementWiseType> : BaseLayer
			where ElementWiseType : ElementWises.IElementWise, new()
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
			/// ElementWiseのサイズ
			/// </summary>
			private int _elem_size;

			/// <summary>
			/// ElementWise
			/// </summary>
			ElementWiseType _elemwize = new ElementWiseType();

			public ElemWiseLayer(int in_height, int in_width, int in_depth,
				int elem_size = 2, int stride = 2, string layer_name = "")
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

				// ElementWise のサイズ
				_elem_size = elem_size;

				// 出力
				_out_hei = _in_hei; _out_wid = _in_wid; _out_dep = (_in_dep - _elem_size) / stride + 1;
				_out_size = _out_hei * _out_wid * _out_dep;
				_outputs = new Matrix<double>[_out_dep];
				for (int i = 0; i < _out_dep; i++)
				{
					_outputs[i] = Matrix<double>.Build.Dense(_out_hei, _out_wid, 0);
				}

				LayerType = "ElementWiseLayer";
				GenericsType = _elemwize.Type();

				_stride = stride;
			}

			public override void ForwardPropagation() { ElementWising(); }

			/// <summary>
			/// BP (ElementWise ver.)
			/// </summary>
			/// <param name="next_delta">
			/// 次層から来た誤差信号 (Σ δ_[l+1] * w_[l+1])
			/// </param>
			/// <returns>前層へ伝播する誤差信号 (Σ δ_[l] * w_[l])</returns>
			public override Vector<double> BackPropagation(Vector<double> next_delta)
			{
				Matrix<double>[] curt_delta = Tools.Converters.ToMatrices(next_delta, _out_dep, _out_hei, _out_wid);

				// 前層に伝播する wdelta
				Matrix<double>[] prev_delta = new Matrix<double>[_in_dep];
				for (int i = 0; i < _in_dep; i++) { prev_delta[i] = Matrix<double>.Build.Dense(_in_hei, _in_wid, 0); }

				int id = 0;
				
				for (int od = 0; od < _out_dep; od++)
				{
					var _df_delta = _inputs.Skip(id).Take(_elem_size);

					for (int oh = 0; oh < _out_hei; oh++)
					{
						for (int ow = 0; ow < _out_wid; ow++)
						{
							// φ'(_output) 計算
							var _df = _elemwize.df(_df_delta.Select(_ => _[oh, ow]).ToArray());

							// δ_[j,h,w] = wdelta_[k,h,w] * φ'(_output)
							for (int es = 0; es < _elem_size; es++)
							{
								prev_delta[id + es][oh, ow] += _df[es] * curt_delta[od][oh, ow];
							}
						}
					}
					id += _stride;
				}

				return Tools.Converters.ToVector(prev_delta, _in_hei, _in_wid);
			}

			/// <summary>
			/// Element wise
			/// </summary>
			/// <returns></returns>
			private void ElementWising()
			{
				int od = 0;
				for (int id = 0; id <= _in_dep - _elem_size; id += _stride)
				{
					// 出力層 _outputs[od] に接続される特徴マップを得る
					var elem_inputs = _inputs.Skip(id).Take(_elem_size).ToArray();

					// 出力計算
					for (int ih = 0; ih < _in_hei; ih++)
					{
						for (int iw = 0; iw < _in_wid; iw++)
						{
							_outputs[od][ih, iw] = _elemwize.f(elem_inputs, ih, iw);
						}
					}
					od++;
				}
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
									_res.Append(_inputs[id][ih, iw] + "\t");
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
							"ElementWiseSize:" + _elem_size + ", " + "Stride:" + _stride);
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
