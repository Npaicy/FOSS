-- start query 1 in stream 5 using template query12.tpl
select  i_item_id
      ,i_item_desc 
      ,i_category 
      ,i_class 
      ,i_current_price
      ,sum(ws_ext_sales_price) as itemrevenue 
      ,sum(ws_ext_sales_price)*100/sum(sum(ws_ext_sales_price)) over
          (partition by i_class) as revenueratio
from	
	web_sales
    	,item 
    	,date_dim
where 
	web_sales.ws_item_sk = item.i_item_sk 
  	and item.i_category in ('Sports', 'Women', 'Books')
  	and web_sales.ws_sold_date_sk = date_dim.d_date_sk
	and date_dim.d_date between cast('1998-02-03' as date) 
				and (cast('1998-02-03' as date) + '30 days'::interval)
group by 
	i_item_id
        ,i_item_desc 
        ,i_category
        ,i_class
        ,i_current_price
order by 
	i_category
        ,i_class
        ,i_item_id
        ,i_item_desc
        ,revenueratio
limit 100;

