/** @file
 *****************************************************************************

 Implementation of interfaces for initializing MNT4.

 See mnt4753_init.hpp .

 *****************************************************************************
 * @author     This file is part of libff, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_g1.hpp>
#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_g2.hpp>
#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_init.hpp>

namespace libff {

// bigint<mnt4753_r_limbs> mnt4753_modulus_r = mnt46_modulus_A;
// bigint<mnt4753_q_limbs> mnt4753_modulus_q = mnt46_modulus_B;

mnt4753_Fq2 mnt4753_twist;
mnt4753_Fq2 mnt4753_twist_coeff_a;
mnt4753_Fq2 mnt4753_twist_coeff_b;
mnt4753_Fq mnt4753_twist_mul_by_a_c0;
mnt4753_Fq mnt4753_twist_mul_by_a_c1;
mnt4753_Fq mnt4753_twist_mul_by_b_c0;
mnt4753_Fq mnt4753_twist_mul_by_b_c1;
mnt4753_Fq mnt4753_twist_mul_by_q_X;
mnt4753_Fq mnt4753_twist_mul_by_q_Y;

bigint<mnt4753_q_limbs> mnt4753_ate_loop_count;
bool mnt4753_ate_is_loop_count_neg;
bigint<4*mnt4753_q_limbs> mnt4753_final_exponent;
bigint<mnt4753_q_limbs> mnt4753_final_exponent_last_chunk_abs_of_w0;
bool mnt4753_final_exponent_last_chunk_is_w0_neg;
bigint<mnt4753_q_limbs> mnt4753_final_exponent_last_chunk_w1;

void init_mnt4753_params()
{
    typedef bigint<mnt4753_r_limbs> bigint_r;
    typedef bigint<mnt4753_q_limbs> bigint_q;

    assert(sizeof(mp_limb_t) == 8 || sizeof(mp_limb_t) == 4); // Montgomery assumes this

    /* parameters for scalar field Fr */
    mnt4753_modulus_r = bigint_r("41898490967918953402344214791240637128170709919953949071783502921025352812571106773058893763790338921418070971888458477323173057491593855069696241854796396165721416325350064441470418137846398469611935719059908164220784476160001");
    assert(mnt4753_Fr::modulus_is_valid());
    if (sizeof(mp_limb_t) == 8)
    {
        mnt4753_Fr::Rsquared = bigint_r("528294247489794478830892053472659225778251862527177273996458536178202249935611532598626300863882178545714358789568805911817755538413250244866201145158388475219888075448690235278457757610027799964891022754525136858589282224337");
        mnt4753_Fr::Rcubed = bigint_r("41633340690057738012868713380844953084100062694720651354492878278511227226750918309707975231298706403293537406447047613688567102129544491565970474558910742579333982008790051581932943967272974145862229340445509069597750977867213");
        mnt4753_Fr::inv = 0xc90776e23fffffff;
    }
    if (sizeof(mp_limb_t) == 4)
    {
        mnt4753_Fr::Rsquared = bigint_r("528294247489794478830892053472659225778251862527177273996458536178202249935611532598626300863882178545714358789568805911817755538413250244866201145158388475219888075448690235278457757610027799964891022754525136858589282224337");
        mnt4753_Fr::Rcubed = bigint_r("41633340690057738012868713380844953084100062694720651354492878278511227226750918309707975231298706403293537406447047613688567102129544491565970474558910742579333982008790051581932943967272974145862229340445509069597750977867213");
        mnt4753_Fr::inv = 0x3fffffff;

    }
    mnt4753_Fr::num_bits = 753;
    mnt4753_Fr::euler = bigint_r("20949245483959476701172107395620318564085354959976974535891751460512676406285553386529446881895169460709035485944229238661586528745796927534848120927398198082860708162675032220735209068923199234805967859529954082110392238080000");
    mnt4753_Fr::s = 30;
    mnt4753_Fr::t = bigint_r("39021010480745652133919498688765463538626870065884617224134041854204007249857398469987226430131438115069708760723898631821547688442835449306011425196003537779414482717728302293895201885929702287178426719326440397855625");
    mnt4753_Fr::t_minus_1_over_2 = bigint_r("19510505240372826066959749344382731769313435032942308612067020927102003624928699234993613215065719057534854380361949315910773844221417724653005712598001768889707241358864151146947600942964851143589213359663220198927812");
    mnt4753_Fr::multiplicative_generator = mnt4753_Fr("17");
    mnt4753_Fr::root_of_unity = mnt4753_Fr("5431548564651772770863376209190533321743766006080874345421017090576169920304713950094628043692772801995471539849411522704471393987882883355624697206026582300050878644000631322086989454860102191886653186986980927065212650747291");
    mnt4753_Fr::nqr = mnt4753_Fr("11");
    mnt4753_Fr::nqr_to_t = mnt4753_Fr("2455648623996886823239043525257449186291028346951286533043435811383049212810466344106778185245641952169091724588148805743116830521168785810979147404935397979718632700247111860253397218855647411901871200410401085305824670182498");
    mnt4753_Fr::small_subgroup_defined = false;

    /* parameters for base field Fq */
    mnt4753_modulus_q = bigint_q("41898490967918953402344214791240637128170709919953949071783502921025352812571106773058893763790338921418070971888253786114353726529584385201591605722013126468931404347949840543007986327743462853720628051692141265303114721689601");
    assert(mnt4753_Fq::modulus_is_valid());
    if (sizeof(mp_limb_t) == 8)
      {
        mnt4753_Fq::Rsquared = bigint_q("3905329767828791615502162626850211818655714094458690120430986575792677496633802118724780745619552999481121157852051288948780064150889938934414916554522736667272858589954728849781314241493825915233155383114266348177232527200456");
        mnt4753_Fq::Rcubed = bigint_q("37585233426682165294065702410202415171476938145667339674403960805833165033715513237837083555904292153853777922114133104918156213519695206603971179794826432236755764454444622189193631114536609372479504766361480176984626948994196");
        mnt4753_Fq::inv = 0xf2044cfbe45e7fff;
      }
    if (sizeof(mp_limb_t) == 4)
      {
        mnt4753_Fq::Rsquared = bigint_q("3905329767828791615502162626850211818655714094458690120430986575792677496633802118724780745619552999481121157852051288948780064150889938934414916554522736667272858589954728849781314241493825915233155383114266348177232527200456");
        mnt4753_Fq::Rcubed = bigint_q("37585233426682165294065702410202415171476938145667339674403960805833165033715513237837083555904292153853777922114133104918156213519695206603971179794826432236755764454444622189193631114536609372479504766361480176984626948994196");
        mnt4753_Fq::inv = 0xe45e7fff;
      }
    mnt4753_Fq::num_bits = 753;
    mnt4753_Fq::euler = bigint_q("20949245483959476701172107395620318564085354959976974535891751460512676406285553386529446881895169460709035485944126893057176863264792192600795802861006563234465702173974920271503993163871731426860314025846070632651557360844800");
    mnt4753_Fq::s = 15;
    mnt4753_Fq::t = bigint_q("1278640471433073529124274133033466709233725278318907137200424283478556909563327233064541435662546964154604216671394463687571830033251476599169665701965732619291119517454523942352538645255842982596454713491581459512424155325");
    mnt4753_Fq::t_minus_1_over_2 = bigint_q("639320235716536764562137066516733354616862639159453568600212141739278454781663616532270717831273482077302108335697231843785915016625738299584832850982866309645559758727261971176269322627921491298227356745790729756212077662");
    mnt4753_Fq::multiplicative_generator = mnt4753_Fq("17");
    mnt4753_Fq::root_of_unity = mnt4753_Fq("40577822398412982719876671814347622311725878559400100565221223860226396934830112376659822430317692232440883010225033880793828874730711721234325694240460855741763791540474706150170374090550695427806583236301930157866709353840964");
    mnt4753_Fq::nqr = mnt4753_Fq("13");
    mnt4753_Fq::nqr_to_t = mnt4753_Fq("18017781208150291467483603956987567350418056191877119616553128888924963195174389973716060278346059199161001506497199268678284023657157367321346359874121888331150637273273499500012599908057395792036980306166250353261196001983326");
    mnt4753_Fq::small_subgroup_defined = false;

    /* parameters for twist field Fq2 */
    mnt4753_Fq2::euler = bigint<2*mnt4753_q_limbs>("877741772694393058372135237733343629593473856016002229857105035140194750058493748062049287411694733142989075570064889940236970783493032261937016406277000694377203416101839277893740346880847050301763334478688551750308947816122818534553695997573207707013723499671859487825892935302013472750294519116226330295520189278343056719919731322734884690958148824900524215249232306591740915846271438433919136803359438630765991579935915053953640737747135454095769600");
    mnt4753_Fq2::s = 16;
    mnt4753_Fq2::t = bigint<2*mnt4753_q_limbs>("26786553121777131908329322440592762133589900391113349299838410496221763612624931276307656476186973057342195909730984190070708336898591072446808362007965109081335553469904763119315806484400849923759867385213884025583158807865076249223440429613440176605643417348384383783749174050964766624459671603888742989975591713816621604001456644370571432219181787869278693092322763262687405879097639112363254907329084430870544176633786470152393821342380842715325");
    mnt4753_Fq2::t_minus_1_over_2 = bigint<2*mnt4753_q_limbs>("13393276560888565954164661220296381066794950195556674649919205248110881806312465638153828238093486528671097954865492095035354168449295536223404181003982554540667776734952381559657903242200424961879933692606942012791579403932538124611720214806720088302821708674192191891874587025482383312229835801944371494987795856908310802000728322185285716109590893934639346546161381631343702939548819556181627453664542215435272088316893235076196910671190421357662");
    mnt4753_Fq2::non_residue = mnt4753_Fq("13");
    mnt4753_Fq2::nqr = mnt4753_Fq2(mnt4753_Fq("8"),mnt4753_Fq("1"));
    mnt4753_Fq2::nqr_to_t = mnt4753_Fq2(mnt4753_Fq("0"),mnt4753_Fq("18228376547502796301079156126471485588453635788107513320149028052132560276470102392885009336437739088956604407962162420327485713003335768904427163750664343990955988139858371693866291895311721436398000747370090550705505161050692"));
    mnt4753_Fq2::Frobenius_coeffs_c1[0] = mnt4753_Fq("1");
    mnt4753_Fq2::Frobenius_coeffs_c1[1] = mnt4753_Fq("41898490967918953402344214791240637128170709919953949071783502921025352812571106773058893763790338921418070971888253786114353726529584385201591605722013126468931404347949840543007986327743462853720628051692141265303114721689600");

    /* parameters for Fq4 */
    mnt4753_Fq4::non_residue = mnt4753_Fq("13");
    mnt4753_Fq4::Frobenius_coeffs_c1[0] = mnt4753_Fq("1");
    mnt4753_Fq4::Frobenius_coeffs_c1[1] = mnt4753_Fq("18691656569803771296244054523431852464958959799019013859007259692542121208304602539555350517075508287829753932558576476751900235650227380562700444433662761577027341858128610410779088384480737679672900770810745291515010467307990");
    mnt4753_Fq4::Frobenius_coeffs_c1[2] = mnt4753_Fq("41898490967918953402344214791240637128170709919953949071783502921025352812571106773058893763790338921418070971888253786114353726529584385201591605722013126468931404347949840543007986327743462853720628051692141265303114721689600");
    mnt4753_Fq4::Frobenius_coeffs_c1[3] = mnt4753_Fq("23206834398115182106100160267808784663211750120934935212776243228483231604266504233503543246714830633588317039329677309362453490879357004638891167538350364891904062489821230132228897943262725174047727280881395973788104254381611");

    /* choice of short Weierstrass curve and its twist */
    mnt4753_G1::coeff_a = mnt4753_Fq("2");
    mnt4753_G1::coeff_b = mnt4753_Fq("28798803903456388891410036793299405764940372360099938340752576406393880372126970068421383312482853541572780087363938442377933706865252053507077543420534380486492786626556269083255657125025963825610840222568694137138741554679540");
    mnt4753_twist = mnt4753_Fq2(mnt4753_Fq::zero(), mnt4753_Fq::one());
    mnt4753_twist_coeff_a = mnt4753_Fq2(mnt4753_G1::coeff_a * mnt4753_Fq2::non_residue, mnt4753_Fq::zero());
    mnt4753_twist_coeff_b = mnt4753_Fq2(mnt4753_Fq::zero(), mnt4753_G1::coeff_b * mnt4753_Fq2::non_residue);
    mnt4753_G2::twist = mnt4753_twist;
    mnt4753_G2::coeff_a = mnt4753_twist_coeff_a;
    mnt4753_G2::coeff_b = mnt4753_twist_coeff_b;
    mnt4753_twist_mul_by_a_c0 = mnt4753_G1::coeff_a * mnt4753_Fq2::non_residue;
    mnt4753_twist_mul_by_a_c1 = mnt4753_G1::coeff_a * mnt4753_Fq2::non_residue;
    mnt4753_twist_mul_by_b_c0 = mnt4753_G1::coeff_b * mnt4753_Fq2::non_residue.squared();
    mnt4753_twist_mul_by_b_c1 = mnt4753_G1::coeff_b * mnt4753_Fq2::non_residue;
    mnt4753_twist_mul_by_q_X = mnt4753_Fq("41898490967918953402344214791240637128170709919953949071783502921025352812571106773058893763790338921418070971888253786114353726529584385201591605722013126468931404347949840543007986327743462853720628051692141265303114721689600");
    mnt4753_twist_mul_by_q_Y = mnt4753_Fq("18691656569803771296244054523431852464958959799019013859007259692542121208304602539555350517075508287829753932558576476751900235650227380562700444433662761577027341858128610410779088384480737679672900770810745291515010467307990");

    /* choice of group G1 */
    mnt4753_G1::G1_zero = mnt4753_G1(mnt4753_Fq::zero(),
                               mnt4753_Fq::one(),
                               mnt4753_Fq::zero());


    mnt4753_G1::G1_one = mnt4753_G1(mnt4753_Fq("23803503838482697364219212396100314255266282256287758532210460958670711284501374254909249084643549104668878996224193897061976788052185662569738774028756446662400954817676947337090686257134874703224133183061214213216866019444443"),
                              mnt4753_Fq("21091012152938225813050540665280291929032924333518476279110711148670464794818544820522390295209715531901248676888544060590943737249563733104806697968779796610374994498702698840169538725164956072726942500665132927942037078135054"),
                              mnt4753_Fq::one());

    mnt4753_G1::wnaf_window_table.resize(0);
    mnt4753_G1::wnaf_window_table.push_back(11);
    mnt4753_G1::wnaf_window_table.push_back(24);
    mnt4753_G1::wnaf_window_table.push_back(60);
    mnt4753_G1::wnaf_window_table.push_back(127);

    mnt4753_G1::fixed_base_exp_window_table.resize(0);
    // window 1 is unbeaten in [-inf, 5.09]
    mnt4753_G1::fixed_base_exp_window_table.push_back(1);
    // window 2 is unbeaten in [5.09, 9.64]
    mnt4753_G1::fixed_base_exp_window_table.push_back(5);
    // window 3 is unbeaten in [9.64, 24.79]
    mnt4753_G1::fixed_base_exp_window_table.push_back(10);
    // window 4 is unbeaten in [24.79, 60.29]
    mnt4753_G1::fixed_base_exp_window_table.push_back(25);
    // window 5 is unbeaten in [60.29, 144.37]
    mnt4753_G1::fixed_base_exp_window_table.push_back(60);
    // window 6 is unbeaten in [144.37, 344.90]
    mnt4753_G1::fixed_base_exp_window_table.push_back(144);
    // window 7 is unbeaten in [344.90, 855.00]
    mnt4753_G1::fixed_base_exp_window_table.push_back(345);
    // window 8 is unbeaten in [855.00, 1804.62]
    mnt4753_G1::fixed_base_exp_window_table.push_back(855);
    // window 9 is unbeaten in [1804.62, 3912.30]
    mnt4753_G1::fixed_base_exp_window_table.push_back(1805);
    // window 10 is unbeaten in [3912.30, 11264.50]
    mnt4753_G1::fixed_base_exp_window_table.push_back(3912);
    // window 11 is unbeaten in [11264.50, 27897.51]
    mnt4753_G1::fixed_base_exp_window_table.push_back(11265);
    // window 12 is unbeaten in [27897.51, 57596.79]
    mnt4753_G1::fixed_base_exp_window_table.push_back(27898);
    // window 13 is unbeaten in [57596.79, 145298.71]
    mnt4753_G1::fixed_base_exp_window_table.push_back(57597);
    // window 14 is unbeaten in [145298.71, 157204.59]
    mnt4753_G1::fixed_base_exp_window_table.push_back(145299);
    // window 15 is unbeaten in [157204.59, 601600.62]
    mnt4753_G1::fixed_base_exp_window_table.push_back(157205);
    // window 16 is unbeaten in [601600.62, 1107377.25]
    mnt4753_G1::fixed_base_exp_window_table.push_back(601601);
    // window 17 is unbeaten in [1107377.25, 1789646.95]
    mnt4753_G1::fixed_base_exp_window_table.push_back(1107377);
    // window 18 is unbeaten in [1789646.95, 4392626.92]
    mnt4753_G1::fixed_base_exp_window_table.push_back(1789647);
    // window 19 is unbeaten in [4392626.92, 8221210.60]
    mnt4753_G1::fixed_base_exp_window_table.push_back(4392627);
    // window 20 is unbeaten in [8221210.60, 42363731.19]
    mnt4753_G1::fixed_base_exp_window_table.push_back(8221211);
    // window 21 is never the best
    mnt4753_G1::fixed_base_exp_window_table.push_back(0);
    // window 22 is unbeaten in [42363731.19, inf]
    mnt4753_G1::fixed_base_exp_window_table.push_back(42363731);

    /* choice of group G2 */
    mnt4753_G2::G2_zero = mnt4753_G2(mnt4753_Fq2::zero(),
                               mnt4753_Fq2::one(),
                               mnt4753_Fq2::zero());

    mnt4753_G2::G2_one = mnt4753_G2(mnt4753_Fq2(mnt4753_Fq("22367666623321080720060256844679369841450849258634485122226826668687008928557241162389052587294939105987791589807198701072089850184203060629036090027206884547397819080026926412256978135536735656049173059573120822105654153939204"), mnt4753_Fq("19674349354065582663569886390557105215375764356464013910804136534831880915742161945711267871023918136941472003751075703860943205026648847064247080124670799190998395234694182621794580160576822167228187443851233972049521455293042")),
                              mnt4753_Fq2(mnt4753_Fq("6945425020677398967988875731588951175743495235863391886533295045397037605326535330657361771765903175481062759367498970743022872494546449436815843306838794729313050998681159000579427733029709987073254733976366326071957733646574"), mnt4753_Fq("17406100775489352738678485154027036191618283163679980195193677896785273172506466216232026037788788436442188057889820014276378772936042638717710384987239430912364681046070625200474931975266875995282055499803236813013874788622488")),
                              mnt4753_Fq2::one());

    mnt4753_G2::wnaf_window_table.resize(0);
    mnt4753_G2::wnaf_window_table.push_back(5);
    mnt4753_G2::wnaf_window_table.push_back(15);
    mnt4753_G2::wnaf_window_table.push_back(39);
    mnt4753_G2::wnaf_window_table.push_back(109);

    mnt4753_G2::fixed_base_exp_window_table.resize(0);
    // window 1 is unbeaten in [-inf, 4.17]
    mnt4753_G2::fixed_base_exp_window_table.push_back(1);
    // window 2 is unbeaten in [4.17, 10.12]
    mnt4753_G2::fixed_base_exp_window_table.push_back(4);
    // window 3 is unbeaten in [10.12, 24.65]
    mnt4753_G2::fixed_base_exp_window_table.push_back(10);
    // window 4 is unbeaten in [24.65, 60.03]
    mnt4753_G2::fixed_base_exp_window_table.push_back(25);
    // window 5 is unbeaten in [60.03, 143.16]
    mnt4753_G2::fixed_base_exp_window_table.push_back(60);
    // window 6 is unbeaten in [143.16, 344.73]
    mnt4753_G2::fixed_base_exp_window_table.push_back(143);
    // window 7 is unbeaten in [344.73, 821.24]
    mnt4753_G2::fixed_base_exp_window_table.push_back(345);
    // window 8 is unbeaten in [821.24, 1793.92]
    mnt4753_G2::fixed_base_exp_window_table.push_back(821);
    // window 9 is unbeaten in [1793.92, 3919.59]
    mnt4753_G2::fixed_base_exp_window_table.push_back(1794);
    // window 10 is unbeaten in [3919.59, 11301.46]
    mnt4753_G2::fixed_base_exp_window_table.push_back(3920);
    // window 11 is unbeaten in [11301.46, 18960.09]
    mnt4753_G2::fixed_base_exp_window_table.push_back(11301);
    // window 12 is unbeaten in [18960.09, 44198.62]
    mnt4753_G2::fixed_base_exp_window_table.push_back(18960);
    // window 13 is unbeaten in [44198.62, 150799.57]
    mnt4753_G2::fixed_base_exp_window_table.push_back(44199);
    // window 14 is never the best
    mnt4753_G2::fixed_base_exp_window_table.push_back(0);
    // window 15 is unbeaten in [150799.57, 548694.81]
    mnt4753_G2::fixed_base_exp_window_table.push_back(150800);
    // window 16 is unbeaten in [548694.81, 1051769.08]
    mnt4753_G2::fixed_base_exp_window_table.push_back(548695);
    // window 17 is unbeaten in [1051769.08, 2023925.59]
    mnt4753_G2::fixed_base_exp_window_table.push_back(1051769);
    // window 18 is unbeaten in [2023925.59, 3787108.68]
    mnt4753_G2::fixed_base_exp_window_table.push_back(2023926);
    // window 19 is unbeaten in [3787108.68, 7107480.30]
    mnt4753_G2::fixed_base_exp_window_table.push_back(3787109);
    // window 20 is unbeaten in [7107480.30, 38760027.14]
    mnt4753_G2::fixed_base_exp_window_table.push_back(7107480);
    // window 21 is never the best
    mnt4753_G2::fixed_base_exp_window_table.push_back(0);
    // window 22 is unbeaten in [38760027.14, inf]
    mnt4753_G2::fixed_base_exp_window_table.push_back(38760027);

    /* pairing parameters */
    mnt4753_ate_loop_count = bigint_q("204691208819330962009469868104636132783269696790011977400223898462431810102935615891307667367766898917669754470400");
    mnt4753_ate_is_loop_count_neg = true;
    mnt4753_final_exponent = bigint<4*mnt4753_q_limbs>("73552111470802397192299133782080682301728710523587802164414953272757803714910813694725910843025422137965798141904448425397132210312763036419196981551382130855705368355580393262211100095907456271531280742739919708794230272306800896198050256355512255795343046414500439648235407402928016221629661971660368018858492377211675996627011913832155809286572006511506918479348970121218134056996473102963627909657625079190739882316882751992741238799066378820181352081085141743775089602078041985556107852922590029377522580702957164527112688206145822971278968699082020672631957410786162945929223941353438866102009621402205679750863679130426460044792078113778548067020007452390228240608175718400");
    mnt4753_final_exponent_last_chunk_abs_of_w0 = bigint_q("204691208819330962009469868104636132783269696790011977400223898462431810102935615891307667367766898917669754470399");
    mnt4753_final_exponent_last_chunk_is_w0_neg = true;
    mnt4753_final_exponent_last_chunk_w1 = bigint_q("1");
}

} // libff
