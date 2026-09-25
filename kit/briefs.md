# Public system briefs

Observations are noisy. Reset randomizes only observables; unobserved quantities use a fixed deterministic initialization rule, which can distribute observed initial totals among hidden compartments without independently randomizing their proportions. There is no random prehistory. Internal variables and equations are not disclosed. These bounds apply to learning and forecasting.

## epidemic

Daily infection onsets and occupied hospital beds in three interacting age groups. School closure changes where contacts occur; masks reduce exposure; vaccination uses a shared clinic workforce. Age groups differ in contacts, severity and recovery.

Clinical referrals can wait for beds, and hospital pressure reduces clinic availability. Behavior, developing immunity and postponed gatherings may retain intervention history. Compare closure with masking at similar case counts, and vaccination before versus after a restriction pulse; follow both cases and hospital recovery.

Unobserved resident compartments start in a fixed age mix determined by the observed initial cases; waiting lists and intervention histories start empty.

Observables: daily_cases, hospital_load.

| Intervention | Minimum | Maximum |
|---|---:|---:|
| school_closure | 0.0 | 1.0 |
| mask_mandate | 0.0 | 1.0 |
| vaccination_rate | 0.0 | 0.003 |

Reference recovery action: `{"mask_mandate": 0.0, "school_closure": 0.0, "vaccination_rate": 0.0}`. This consumes paid steps and does not reset the system.

Reference pulse action: `{"mask_mandate": 1.0, "school_closure": 1.0, "vaccination_rate": 0.003}`. Recovery scenarios use 70–100% of the distance from recovery to this action, independently per control.

## market

Price, traded volume and order-book depth. Interest rate and transaction tax are fractions per trading period. Producer and consumer groups have different reservation values, storage capacities and placement speeds.

Production consumes working cash; final consumption returns revenue. Their orders move through preparation and execution; a new policy does not cancel commitments already made. Completed trades transfer goods and cash between customers and finite dealer books.

Dealers share settlement and outside hedging resources. Inventory may tie up funding until settlement, adverse price moves may reduce risk capacity, and investors may shift desired exposure toward recently successful strategies. Reported depth is an aggregate: the side, location and settlement status of the available capacity can change what a subsequent reversal does.

Customer warehouses start half full with full working cash, dealer books start neutral, and orders and settlement commitments start empty.

Observables: price, volume, depth.

| Intervention | Minimum | Maximum |
|---|---:|---:|
| interest_rate | 0.0 | 0.1 |
| transaction_tax | 0.0 | 0.05 |

Reference recovery action: `{"interest_rate": 0.0, "transaction_tax": 0.0}`. This consumes paid steps and does not reset the system.

Reference pulse action: `{"interest_rate": 0.1, "transaction_tax": 0.05}`. Recovery scenarios use 70–100% of the distance from recovery to this action, independently per control.

## traffic

Observe terminal flows and mean speeds on two routes. Light and heavy vehicles retain their route, class and crossing commitments after entry. Toll changes the arriving mix; ramp metering changes admitted demand.

Signal timing divides new crossing admissions, freight priority changes which waiting class gets service, and clearance effort shifts a shared crew from intersection operation toward downstream exits. Vehicles already crossing keep occupying the shared junction until their committed work finishes and exit space opens. Thus one route can obstruct the other even under unchanged signals.

Lane closure affects different sections unequally. Finite approach, junction and exit buffers reject excess arrivals; waiting approach drivers may divert. Route learning, crew fatigue/switching costs and persistent spillback fronts are three possible mechanisms; exactly two apply.

Compare toll and ramp preparations with similar totals to change vehicle mix; compare clearance with a signal reversal after stopping arrivals. Reported speed combines observed completed journey times with current stopped and moving class mix; equal route totals can therefore report different speeds. Roads and crossings start empty.

Observables: flow_a, flow_b, speed_a, speed_b.

| Intervention | Minimum | Maximum |
|---|---:|---:|
| signal_timing | 0.1 | 0.9 |
| lane_closure | 0.0 | 0.75 |
| toll | 0.0 | 5.0 |
| ramp_metering | 0.0 | 1.0 |
| freight_priority | 0.0 | 1.0 |
| clearance_effort | 0.0 | 1.0 |

Reference recovery action: `{"clearance_effort": 1.0, "freight_priority": 0.5, "lane_closure": 0.0, "ramp_metering": 0.0, "signal_timing": 0.5, "toll": 5.0}`. This consumes paid steps and does not reset the system.

Reference pulse action: `{"clearance_effort": 0.0, "freight_priority": 1.0, "lane_closure": 0.65, "ramp_metering": 1.0, "signal_timing": 0.15, "toll": 0.0}`. Recovery scenarios use 70–100% of the distance from recovery to this action, independently per control.

## power_grid

Grid load in power units, frequency in Hz, and renewable share of delivered generation. Price signal reduces desired demand; reserve dispatch requests additional supply in power units. A tick is one dispatch interval.

Supply requests can be limited by available physical resources. Reserve resources differ in power, duration and thermal response, and share a charging connection. Dispatch allocates supply according to operating cost and system conditions.

Flexible cooling loads have different thermal response times. Price shifts their thermostat settings: each load warms while off, cools while on, and switches at separate upper and lower temperature limits. Aggregate consumption depends on the distribution of temperatures and which loads are already running.

A price pulse can synchronize some loads and cause a later rebound; thermal heterogeneity disperses that synchronization. Every reset starts with the same asynchronous population at reference price 0.8, without random hidden phases. Charging allowance limits grid power available for refilling reserves.

Interconnector setting opens remote delivery capacity, whose temperature depends on recent flows. Renewables can be curtailed at that connection. Conventional governors respond to frequency with finite response times and output limits.

Observables: load, frequency, renewable_share.

| Intervention | Minimum | Maximum |
|---|---:|---:|
| price_signal | 0.0 | 2.0 |
| reserve_dispatch | 0.0 | 150.0 |
| charging_allowance | 0.0 | 1.0 |
| interconnector | 0.0 | 1.0 |

Reference recovery action: `{"charging_allowance": 1.0, "interconnector": 1.0, "price_signal": 1.5, "reserve_dispatch": 0.0}`. This consumes paid steps and does not reset the system.

Reference pulse action: `{"charging_allowance": 0.0, "interconnector": 0.2, "price_signal": 0.0, "reserve_dispatch": 150.0}`. Recovery scenarios use 70–100% of the distance from recovery to this action, independently per control.

## supply_chain

Arrivals at the retailer plus total supplier and retail stock. Supplier goods have two classes; product mix selects new production and requested dispatch, without rewriting old batches. Orders withdraw only available stock.

Production effort releases material and operates the primary line; receiving effort staffs the final terminal. Rush handling bypasses treatment for new primary-line departures, while the second class normally follows its own intake. Conveyors retain their destination and travel commitment after a rush change.

The two intakes share forward transport; the second intake and treatment share cooling, while the primary intake, receiving and maintenance share drive service. Maintenance restores treatment activity but takes utility and productive treatment time. Goods needing rework may return to the primary intake; return-space overflow leaves as secondary-grade goods counted in shipments.

Congested transport and rework, machine heat/wear, and adaptive production commitments are three possible memory mechanisms; exactly two apply. Hold product mix fixed when comparing maintenance with an idle pause; compare rush changes before and after dispatch, or equal orders in opposite production sequences. Internal buffers and conveyors start empty; the initial stocks split into fixed class shares.

Observables: shipments, inventory_supplier, inventory_retail.

| Intervention | Minimum | Maximum |
|---|---:|---:|
| order_quantity | 0.0 | 80.0 |
| lead_time_buy | 0.0 | 1.0 |
| product_mix | 0.0 | 1.0 |
| production_effort | 0.0 | 1.5 |
| receiving_effort | 0.0 | 1.5 |
| maintenance | 0.0 | 1.0 |

Reference recovery action: `{"lead_time_buy": 1.0, "maintenance": 1.0, "order_quantity": 0.0, "product_mix": 0.5, "production_effort": 1.0, "receiving_effort": 1.5}`. This consumes paid steps and does not reset the system.

Reference pulse action: `{"lead_time_buy": 0.2, "maintenance": 0.0, "order_quantity": 80.0, "product_mix": 0.8, "production_effort": 1.5, "receiving_effort": 0.35}`. Recovery scenarios use 70–100% of the distance from recovery to this action, independently per control.

## wildlife

Observe prey and predator totals in northern and southern landscapes. Hunting requests prey harvest; habitat protection changes shelter and resource renewal, especially in the north. Corridor access controls new interregional journeys; already travelling animals can still arrive.

Within each region, open pasture, mixed cover and sheltered browse differ in feeding, hunting exposure and predation. Food renewal shares a finite resource, young animals compete for nursery food, and arrivals compete for settlement space. Compare habitat recovery with corridors closed and open, and harvest before versus after protection.

Regional totals alone do not identify patch occupancy, juvenile condition or animals in transit.

Observables: prey_north, predator_north, prey_south, predator_south.

| Intervention | Minimum | Maximum |
|---|---:|---:|
| hunting_quota | 0.0 | 8.0 |
| habitat_protection | 0.0 | 1.0 |
| corridor_access | 0.0 | 1.0 |

Reference recovery action: `{"corridor_access": 0.0, "habitat_protection": 1.0, "hunting_quota": 0.0}`. This consumes paid steps and does not reset the system.

Reference pulse action: `{"corridor_access": 1.0, "habitat_protection": 0.1, "hunting_quota": 7.0}`. Recovery scenarios use 70–100% of the distance from recovery to this action, independently per control.

## reservoir

Reservoir water volume, incoming and delivered outgoing water per tick, and an outlet water-quality index from zero to one. Release and irrigation request water per tick. Withdrawal depth selects shallow versus deep release; irrigation draws near the surface.

Aeration increases oxygen transfer and vertical mixing. Water is stratified: temperature, dissolved material, algae and oxygen can differ by depth, although only aggregate volume and outlet quality are observed. Seasonal river supply brings nutrients.

Light, nutrient availability, decomposition and oxygen interact. Biomass can foul withdrawal screens, restricting actual discharge; flushing and aeration remove some of that attached material. Groundwater, irrigated land and deposited material may return water or contaminants after a delay.

A deep release can change later surface quality by changing stored layers, while aeration can either dilute an outlet or remobilize deeper material. Initial instrument readings are observed; internal layers start from the same reference profile on every reset. A tick is one operating day.

Observables: level, inflow, outflow, quality.

| Intervention | Minimum | Maximum |
|---|---:|---:|
| release_rate | 0.0 | 12.0 |
| irrigation_allocation | 0.0 | 8.0 |
| withdrawal_depth | 0.0 | 1.0 |
| aeration | 0.0 | 1.0 |

Reference recovery action: `{"aeration": 1.0, "irrigation_allocation": 0.0, "release_rate": 2.0, "withdrawal_depth": 0.0}`. This consumes paid steps and does not reset the system.

Reference pulse action: `{"aeration": 0.0, "irrigation_allocation": 8.0, "release_rate": 12.0, "withdrawal_depth": 1.0}`. Recovery scenarios use 70–100% of the distance from recovery to this action, independently per control.

## ad_auction

Observe auction win fraction, spend and completed conversions. Choose bid, per-tick budget and targeting breadth. Broader targeting contains narrower audiences, whose members can be in different stages of attention and purchase.

A won impression is not necessarily an immediate conversion: started purchases remain committed and compete for limited fulfillment work. Different audiences require different amounts of fulfillment work. Converted customers take time to become available again.

Rival campaigns may move a shared pool of capital between audiences, repeated exposure may temporarily remove reachable people, and broad introduction may change the effect of later follow-up. Equal spending can therefore leave different future opportunities. Initial reports are the only random initial state; hidden populations and commitments begin from the same reference conditions on every reset: people are available and unprepared, with no pending purchases or exposure recovery.

Observables: win_rate, spend, conversions.

| Intervention | Minimum | Maximum |
|---|---:|---:|
| bid | 0.0 | 5.0 |
| budget_cap | 0.0 | 100.0 |
| targeting_breadth | 0.1 | 1.0 |

Reference recovery action: `{"bid": 1.5, "budget_cap": 20.0, "targeting_breadth": 0.55}`. This consumes paid steps and does not reset the system.

Reference pulse action: `{"bid": 5.0, "budget_cap": 100.0, "targeting_breadth": 0.775}`. Recovery scenarios use 70–100% of the distance from recovery to this action, independently per control.

## social_contagion

Observe adopter totals in two communities. Seeding funds outreach, incentive changes the offer, and bridge outreach allocates effort between local recruitment and introductions across communities. Relationship-led, incentive-led and deliberative audiences mix differently.

Interested people must complete onboarding through a workforce also needed by existing members; promises accompany waiting cohorts. Disappointed former members need time before reconsidering. Credibility, incentive expectations and cross-community relationships can retain history.

Compare local versus bridge campaigns, incentive before versus after recruitment, and recovery with new outreach stopped. Initial members have a fixed disclosed-style community mix and no paid promises; other queues begin empty.

Observables: adopters_a, adopters_b.

| Intervention | Minimum | Maximum |
|---|---:|---:|
| seeding | 0.0 | 10.0 |
| incentive | 0.0 | 2.0 |
| bridge_outreach | 0.0 | 1.0 |

Reference recovery action: `{"bridge_outreach": 0.0, "incentive": 0.0, "seeding": 0.0}`. This consumes paid steps and does not reset the system.

Reference pulse action: `{"bridge_outreach": 0.6, "incentive": 2.0, "seeding": 9.0}`. Recovery scenarios use 70–100% of the distance from recovery to this action, independently per control.

## hospital_queue

Observe estimated wait, patients pending or receiving hospital care, and gross discharges. Routine, urgent and elective cases need different assessment and treatment work. Diagnostic allocation divides shared staff; urgent priority changes new service admissions.

Patients already in assessment or treatment retain their work and occupy finite chairs or beds until completion. A completed assessment may hold its chair when the treatment queue is full. Staffing and overtime change work delivered, not completed patient counts directly.

Overtime can create later fatigue; staff changes can require orientation before staff are fully effective. Follow-up capacity diverts shared staff to a finite outside program that can prevent delayed returns after discharge. Patients waiting can deteriorate or leave; overflow is referred elsewhere rather than stored in an invisible queue.

Fatigue, handover and returning case mix are three possible mechanisms; exactly two apply. Compare equal staff-hours with different overtime spacing, change diagnostic allocation at fixed staffing, or add follow-up after the same discharge burst. Initial queue composition is a fixed function of the reported initial count; services and the follow-up program start empty.

Observables: wait_time, queue, discharges.

| Intervention | Minimum | Maximum |
|---|---:|---:|
| staffing | 1.0 | 20.0 |
| elective_scheduling | 0.0 | 20.0 |
| diagnostic_allocation | 0.1 | 0.8 |
| urgent_priority | 0.0 | 1.0 |
| overtime | 0.0 | 1.0 |
| followup_capacity | 0.0 | 1.0 |

Reference recovery action: `{"diagnostic_allocation": 0.4, "elective_scheduling": 0.0, "followup_capacity": 1.0, "overtime": 0.0, "staffing": 20.0, "urgent_priority": 0.6}`. This consumes paid steps and does not reset the system.

Reference pulse action: `{"diagnostic_allocation": 0.75, "elective_scheduling": 20.0, "followup_capacity": 0.0, "overtime": 1.0, "staffing": 5.0, "urgent_priority": 1.0}`. Recovery scenarios use 70–100% of the distance from recovery to this action, independently per control.

