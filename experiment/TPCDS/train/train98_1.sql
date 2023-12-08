-- start query 1 in stream 0 using template query98.tpl
select i_item_id
      ,i_item_desc 
      ,i_category 
      ,i_class 
      ,i_current_price
      ,sum(ss_ext_sales_price) as itemrevenue 
      ,sum(ss_ext_sales_price)*100/sum(sum(ss_ext_sales_price)) over
          (partition by i_class) as revenueratio
from	
	store_sales
    	,item 
    	,date_dim
where 
	store_sales.ss_item_sk = item.i_item_sk 
  	and item.i_category in ('Jewelry', 'Sports', 'Books')
  	and store_sales.ss_sold_date_sk = date_dim.d_date_sk
	and date_dim.d_date between cast('2001-01-12' as date) and (cast('2001-01-12' as date) + '30 days'::interval)
group by 
	item.i_item_id
        ,item.i_item_desc 
        ,item.i_category
        ,item.i_class
        ,item.i_current_price
order by 
	item.i_category
        ,item.i_class
        ,item.i_item_id
        ,item.i_item_desc
        ,revenueratio;

