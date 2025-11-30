--
-- PostgreSQL database dump
--

\restrict 3h0zAKhBeY9J2ZTj7QtD6TbczTLKBqcVFGxzvOZY7kgsqfNis4a9i2JGwDODouX

-- Dumped from database version 18.1
-- Dumped by pg_dump version 18.1

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: banks_table; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.banks_table (
    bank_id character varying(50) NOT NULL,
    bank_name character varying(100) NOT NULL,
    app_name character varying(100)
);


ALTER TABLE public.banks_table OWNER TO postgres;

--
-- Name: reviews_table; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.reviews_table (
    review_id character varying(50) NOT NULL,
    bank_id character varying(50),
    review_text character varying,
    rating integer,
    review_date date,
    sentiment_label character varying,
    sentiment_score double precision,
    source character varying
);


ALTER TABLE public.reviews_table OWNER TO postgres;

--
-- Name: banks_table banks_table_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.banks_table
    ADD CONSTRAINT banks_table_pkey PRIMARY KEY (bank_id);


--
-- Name: reviews_table reviews_table_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.reviews_table
    ADD CONSTRAINT reviews_table_pkey PRIMARY KEY (review_id);


--
-- Name: reviews_table reviews_table_bank_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.reviews_table
    ADD CONSTRAINT reviews_table_bank_id_fkey FOREIGN KEY (bank_id) REFERENCES public.banks_table(bank_id);


--
-- PostgreSQL database dump complete
--

\unrestrict 3h0zAKhBeY9J2ZTj7QtD6TbczTLKBqcVFGxzvOZY7kgsqfNis4a9i2JGwDODouX

